//! A [variable-length encoding](https://en.wikipedia.org/wiki/Variable-length_encoding) for unsigned 64 bit integers.
//!
//! The core idea of this encoding scheme is to split the encoding into two parts: a *tag* which indicates how many bytes are used to encode the int, and then the actual *int encoding*, encoded in zero (sufficiently small ints can be inlined into the tag), one, two, four, or eight bytes. You can use tags of any width between two and eight bits — the wider the tag, the more int encodings can be inlined into the tag. The advantage of smaller tags is that multiple of them can fit into a single byte.
//!
//! We have a detailed [writeup on the encoding here](https://willowprotocol.org/specs/encodings/index.html#compact_integers). But to keep things self-contained, here is a precise definition of the possible codes for any `u64` `n` and any number `2 <= tag_width <= 8`:
//!
//! - You can use the numerically greatest possible `tag_width`-bit integer as the *tag*, and the eight-byte big-endian encoding of `n` as the *int encoding*.
//! - If `n < 256^4`, you can use the numerically second-greatest possible `tag_width`-bit integer as the *tag*, and the four-byte big-endian encoding of `n` as the *int encoding*.
//! - If `n < 256^2`, you can use the numerically third-greatest possible `tag_width`-bit integer as the *tag*, and the two-byte big-endian encoding of `n` as the *int encoding*.
//! - If `n < 256`, you can use the numerically third-greatest possible `tag_width`-bit integer as the *tag*, and the one-byte encoding of `n` as the *int encoding*.
//! - If `tag_width > 2`, and if `n` is less than the numerically fourth-greatest `tag_width`-bit integer, you can use the `tag_width`-bit encoding of `n` as the *tag*, and the empty string as the `int encoding`.
//!
//! Our implementation uses the [`codec`] module of [ufotofu](https://worm-blossom.org/ufotofu/) to abstract over actual byte storage.
//!
//! ## Encoding API
//!
//! Use [`write_tag`] to write *tags* at arbitrary offsets into any [`u8`], and use [`cu64_encode`] to write the minimal *int encoding* for any given tag width into a [`BulkConsumer<Item = u8>`](BulkConsumer).
//!
//! ```
//! use ufotofu::codec_prelude::*;
//! use compact_u64::*;
//!
//! // Encode two u64s using two four-bit tags, combining the tags into a single byte.
//! # pollster::block_on(async {
//! let n1 = 258; // Requires two bytes for its int encoding.
//! let n2 = 7; // Can be inlined into the tag.
//! let tag_width1 = 4;
//! let tag_offset1 = 0;
//! let tag_width2 = 4;
//! let tag_offset2 = 4;
//!
//! let mut tag_byte = 0;
//! // Write the four-bit tag for `n1` into `tag_byte`, starting at the most significant bit.
//! write_tag(&mut tag_byte, tag_width1, tag_offset1, n1);
//! // Write the four-bit tag for `n2` into `tag_byte`, starting at the fifth-most significant bit.
//! write_tag(&mut tag_byte, tag_width2, tag_offset2, n2);
//!
//! // First four bits indicate a 2-byte int encoding, remaining four bits inline the integer `7`.
//! assert_eq!(tag_byte, 0xd7);
//!
//! // The buffer into which we will write the encodings.
//! let mut buf = [0; 3];
//! let mut con = (&mut buf).into_consumer();
//!
//! // First write the byte containing the two tags.
//! con.consume_item(tag_byte).await;
//!
//! // Writing the int encoding for `n1` will write the two-byte big-endian code for `n1`.
//! cu64_encode(n1, tag_width1, &mut con).await.unwrap();
//! // Writing the int encoding for `n2` is a no-op, because `n2` is inlined in the tag.
//! cu64_encode(n2, tag_width2, &mut con).await.unwrap();
//!
//! assert_eq!(buf, [0xd7, 1, 2]);
//! # Result::<(), ()>::Ok(())
//! # });
//! ```
//!
//! You can further use [`cu64_len_of_encoding`] to determine the number of bytes an *int encoding* would require for a given *tag*.
//!
//! ## Decoding API
//!
//! Use [`cu64_decode`] to decode the *int encoding* for some given *tag* from a [`BulkProducer<Item = u8>`](BulkProducer). Use [`cu64_decode_canonic`] if you want to reject non-minimal encodings.
//!
//! ```
//! use ufotofu::codec_prelude::*;
//! use compact_u64::*;
//!
//! // We will decode a single byte containing two four-bit tags, and then decode
//! // two int encodings corresponding to the two tags.
//! # pollster::block_on(async {
//! let tag_width1 = 4;
//! let tag_offset1 = 0;
//! let tag_width2 = 4;
//! let tag_offset2 = 4;
//!
//! // The encoding of two four-bit tags followed by two int encodings, for ints `258` and `7`.
//! let mut pro = producer::clone_from_slice(&[0xd7, 1, 2][..]);
//!
//! // Read the byte that combines the two tags from the producer.
//! let mut tag_byte = pro.produce_item().await?;
//!
//! // Decode the two ints.
//! let n1 = cu64_decode(tag_byte, tag_width1, tag_offset1, &mut pro).await?;
//! let n2 = cu64_decode(tag_byte, tag_width2, tag_offset2, &mut pro).await?;
//!
//! assert_eq!(n1, 258);
//! assert_eq!(n2, 7);
//! # Result::<(), DecodeError<(), Infallible, Infallible>>::Ok(())
//! # });
//! ```
//!
//! ## Standalone APIs
//!
//! The previous examples demonstrated the APIs for processing *tags* and *int encodings* separately. For the common case where you use an eight-bit *tag* immediately followed by the corresponding *int encoding*, we offer a more convenient API via [`cu64_encode_standalone`] and [`cu64_decode_standalone`] (and [`cu64_decode_canonic_standalone`] for rejecting non-minimal encodings):
//!
//! ```
//! use ufotofu::codec_prelude::*;
//! use compact_u64::*;
//!
//! # pollster::block_on(async {
//! let mut buf = [0; 3];
//! let mut con = (&mut buf).into_consumer();
//!
//! cu64_encode_standalone(258, &mut con).await.unwrap();
//! assert_eq!(buf, [0xfd, 1, 2]);
//!
//! let n = cu64_decode_standalone(&mut producer::clone_from_slice(&buf[..])).await.unwrap();
//! assert_eq!(n, 258);
//! # });
//! ```
//!
//! The same functionality is also exposed through the [`CompactU64`] type, which is a thin wrapper around `u64` that implements the [`Encodable`], [`EncodableKnownLength`], [`Decodable`], and [`DecodableCanonic`] traits.
//!
//! ```
//! use ufotofu::codec_prelude::*;
//! use compact_u64::*;
//!
//! # pollster::block_on(async {
//! let mut buf = [0; 3];
//! let mut con = (&mut buf).into_consumer();
//!
//! con.consume_encoded(&CompactU64(258)).await.unwrap();
//! assert_eq!(buf, [0xfd, 1, 2]);
//!
//! let n: CompactU64 = producer::clone_from_slice(&buf[..]).produce_decoded().await.unwrap();
//! assert_eq!(n.0, 258);
//! # });
//! ```
//!
//! ## Invalid Parameters
//!
//! The [offset of a tag](TagOffset) must always be a number between zero and seven (inclusive), and the [width of a tag](TagWidth) must always be a number between two and eight (inclusive). When a function takes a [`TagWidth`] and a [`TagOffset`], their sum must be at most eight.
//!
//! All functions in this crate may exhibit unspecified (but always safe) behaviour if these invariants are broken. When debug assertions are enabled, all functions in this crate are guaranteed to panic when these invariants are broken.

#![no_std]
use core::fmt::Display;

#[cfg(feature = "dev")]
use arbitrary::Arbitrary;

use ufotofu::codec_prelude::*;

/// The width of a *tag* — between two and eight inclusive.
pub type TagWidth = u8;

#[inline(always)]
const fn assert_tag_width(tag_width: TagWidth) {
    debug_assert!(2 <= tag_width);
    debug_assert!(tag_width <= 8);
}

/// The offset of a *tag* within a [`TagByte`] — between zero (most significant) and seven (least significant).
///
/// Zero indicates the most significant bit, seven indicates the least significant bit.
pub type TagOffset = u8;

#[inline(always)]
const fn assert_tag_offset(tag_offset: TagOffset, tag_width: TagWidth) {
    debug_assert!(tag_offset + tag_width <= 8);
}

/// A byte storing some number of *tags*.
pub type TagByte = u8;

// Always one of 0, 1, 2, 4, or 8.
type EncodingWidth = usize;

// A byte whose least significant bits store a tag, and whose other bits are set to zero.
type Tag = u8;

/// Writes the minimal tag for the given u64 `n` into the `tag_byte`, for a given [`TagWidth`] and at a given [`TagOffset`].
///
/// Invariant: `tag_width + tag_offset <= 8`, else anything (safe) may happen. When debug assertions are enabled, this function will panic if the invariant is broken.
///
/// ```
/// use compact_u64::write_tag;
///
/// let mut tag_byte1 = 0;
/// write_tag(&mut tag_byte1, 3, 2, u64::MAX);
/// assert_eq!(tag_byte1, 0b0011_1000);
///
/// let mut tag_byte2 = 0;
/// write_tag(&mut tag_byte2, 3, 2, 258);
/// assert_eq!(tag_byte2, 0b0010_1000);
///
/// let mut tag_byte3 = 0;
/// write_tag(&mut tag_byte3, 3, 2, 3);
/// assert_eq!(tag_byte3, 0b0001_1000);
/// ```
pub fn write_tag(tag_byte: &mut TagByte, tag_width: TagWidth, tag_offset: TagOffset, n: u64) {
    assert_tag_width(tag_width);
    assert_tag_offset(tag_offset, tag_width);
    debug_assert!(tag_width + tag_offset <= 8);

    let new_tag_byte: TagByte = min_tag(n, tag_width) << (8 - (tag_offset + tag_width));
    *tag_byte |= new_tag_byte;
}

/// Writes the *int encoding* of `n` for the minimal *tag* of width `tag_width` into the `consumer`.
///
/// ```
/// use ufotofu::codec_prelude::*;
/// use compact_u64::*;
/// # pollster::block_on(async {
/// let mut buf = [0; 2];
/// let mut con = (&mut buf).into_consumer();
///
/// cu64_encode(258, 8, &mut con).await.unwrap();
/// assert_eq!(buf, [1, 2]);
/// # Result::<(), ()>::Ok(())
/// # });
/// ```
pub async fn cu64_encode<C>(n: u64, tag_width: TagWidth, consumer: &mut C) -> Result<(), C::Error>
where
    C: BulkConsumer<Item = u8> + ?Sized,
{
    let encoding_width = cu64_len_of_encoding(tag_width, n);

    match encoding_width {
        0 => Ok(()),
        1 => consumer.consume_item(n as u8).await,
        2 => consumer.encode_u16_be(n as u16).await,
        4 => consumer.encode_u32_be(n as u32).await,
        8 => consumer.encode_u64_be(n as u64).await,
        _ => unreachable!(),
    }
}

/// Writes the minimal eight-bit *tag* for `n` and then the corresponding minimal *int encoding* into the `consumer`.
///
/// ```
/// use ufotofu::codec_prelude::*;
/// use compact_u64::*;
///
/// # pollster::block_on(async {
/// let mut buf = [0; 3];
/// let mut con = (&mut buf).into_consumer();
///
/// cu64_encode_standalone(258, &mut con).await.unwrap();
/// assert_eq!(buf, [0xfd, 1, 2]);
/// # });
/// ```
pub async fn cu64_encode_standalone<C>(n: u64, consumer: &mut C) -> Result<(), C::Error>
where
    C: BulkConsumer<Item = u8> + ?Sized,
{
    let tag = min_tag(n, 8);
    consumer.consume_item(tag).await?;
    cu64_encode(n, 8, consumer).await
}

/// Returns the length of the *int encoding* of `n` when using a minimal *tag* of the given `tag_width`.
///
/// ```
/// use ufotofu::codec_prelude::*;
/// use compact_u64::*;
///
/// assert_eq!(cu64_len_of_encoding(8, 111), 0);
/// assert_eq!(cu64_len_of_encoding(8, 254), 1);
/// assert_eq!(cu64_len_of_encoding(8, 258), 2);
/// ```
pub const fn cu64_len_of_encoding(tag_width: TagWidth, n: u64) -> usize {
    min_width(n, tag_width)
}

/// Reads the *int encoding* of `n` for the given *tag* from the `producer`.
///
/// The *tag* of width `tag_width` is read at offset `tag_offset` from the `tag_byte`.
///
/// ```
/// use ufotofu::codec_prelude::*;
/// use compact_u64::*;
/// # pollster::block_on(async {
/// let mut buf = [1, 2];
///
/// let n = cu64_decode(
///     0b0010_1000, // the tag byte, the actual tag being `101` (the outer zeros are ignored)
///     3, // the tag width
///     2, // the tag offset
///     &mut producer::clone_from_slice(&buf[..]),
/// ).await.unwrap();
/// assert_eq!(n, 258);
/// # Result::<(), ()>::Ok(())
/// # });
/// ```
pub async fn cu64_decode<P>(
    tag_byte: TagByte,
    tag_width: TagWidth,
    tag_offset: TagOffset,
    producer: &mut P,
) -> Result<u64, DecodeError<P::Final, P::Error, Infallible>>
where
    P: BulkProducer<Item = u8> + ?Sized,
{
    let tag = extract_tag(tag_byte, tag_width, tag_offset);
    let encoding_width = encoding_width_from_tag(tag, tag_width);

    Ok(match encoding_width {
        0 => tag as u64,
        1 => producer.produce_item().await? as u64,
        2 => producer.decode_u16_be().await? as u64,
        4 => producer.decode_u32_be().await? as u64,
        8 => producer.decode_u64_be().await?,
        _ => unreachable!(),
    })
}

/// Reads an eight-bit *tag* from the `producer`, then reads the corresponding *int encoding* from the `producer`, and returns the decoded [`u64`].
///
/// ```
/// use ufotofu::codec_prelude::*;
/// use compact_u64::*;
///
/// # pollster::block_on(async {
/// let n = cu64_decode_standalone(
///     &mut producer::clone_from_slice(&[0xfd, 1, 2][..]),
/// ).await.unwrap();
///
/// assert_eq!(n, 258);
/// # });
/// ```
pub async fn cu64_decode_standalone<P>(
    producer: &mut P,
) -> Result<u64, DecodeError<P::Final, P::Error, Infallible>>
where
    P: BulkProducer<Item = u8> + ?Sized,
{
    let tag = producer.produce_item().await?;
    cu64_decode(tag, 8, 0, producer).await
}

/// Reads the *int encoding* of `n` for the given *tag* from the `producer`, yielding an error if the encoding was not minimal.
///
/// The *tag* of width `tag_width` is read at offset `tag_offset` from the `tag_byte`.
///
/// ```
/// use ufotofu::codec_prelude::*;
/// use compact_u64::*;
/// # pollster::block_on(async {
/// let mut buf = [1, 2];
///
/// let n = cu64_decode_canonic(
///     0b0010_1000, // the tag byte, the actual tag being `101` (the outer zeros are ignored)
///     3, // the tag width
///     2, // the tag offset
///     &mut producer::clone_from_slice(&buf[..]),
/// ).await.unwrap();
/// assert_eq!(n, 258);
///
/// let mut non_minimal_buf = [0, 0, 1, 2];
///
/// assert!(cu64_decode_canonic(
///     0xfe, // an eight-bit tag indicating an *int encoding* of four bytes
///     8, // the tag width
///     0, // the tag offset
///     &mut producer::clone_from_slice(&buf[..]),
/// ).await.is_err());
/// # Result::<(), ()>::Ok(())
/// # });
/// ```
pub async fn cu64_decode_canonic<P>(
    tag_byte: TagByte,
    tag_width: TagWidth,
    tag_offset: TagOffset,
    producer: &mut P,
) -> Result<u64, DecodeError<P::Final, P::Error, NotMinimal>>
where
    P: BulkProducer<Item = u8> + ?Sized,
{
    let decoded = cu64_decode(tag_byte, tag_width, tag_offset, producer)
        .await
        .map_err(|err| err.map_other(|_| unreachable!()))?;

    if extract_tag(tag_byte, tag_width, tag_offset) == min_tag(decoded, tag_width) {
        Ok(decoded)
    } else {
        Err(DecodeError::Other(NotMinimal))
    }
}

/// Reads an eight-bit *tag* from the `producer`, then reads the corresponding *int encoding* from the `producer`, and returns the decoded [`u64`], or an error if the encoding was not minimal.
///
/// ```
/// use ufotofu::codec_prelude::*;
/// use compact_u64::*;
///
/// # pollster::block_on(async {
/// let n = cu64_decode_canonic_standalone(
///     &mut producer::clone_from_slice(&[0xfd, 1, 2][..]),
/// ).await.unwrap();
/// assert_eq!(n, 258);
///
/// assert!(cu64_decode_canonic_standalone(
///     &mut producer::clone_from_slice(&[0xfe, 0, 0, 1, 2][..]), // int encoding in four bytes
/// ).await.is_err());
/// # });
/// ```
pub async fn cu64_decode_canonic_standalone<P>(
    producer: &mut P,
) -> Result<u64, DecodeError<P::Final, P::Error, NotMinimal>>
where
    P: BulkProducer<Item = u8> + ?Sized,
{
    let tag = producer.produce_item().await?;
    cu64_decode_canonic(tag, 8, 0, producer).await
}

fn extract_tag(tag_byte: TagByte, tag_width: TagWidth, tag_offset: TagOffset) -> Tag {
    assert_tag_width(tag_width);
    assert_tag_offset(tag_offset, tag_width);

    match 8_usize.checked_sub((tag_offset as usize) + (tag_width as usize)) {
        None => panic!("Invalid tag offset: {}", tag_offset),
        Some(shift_by) => {
            let max_tag = maximal_tag(tag_width);

            (tag_byte >> shift_by) & max_tag
        }
    }
}

fn encoding_width_from_tag(tag: Tag, tag_width: TagWidth) -> EncodingWidth {
    match maximal_tag(tag_width) - tag {
        0 => 8,
        1 => 4,
        2 => 2,
        3 => 1,
        _ => 0,
    }
}

/// Returns the least [`EncodingWidth`] a given [`u64`] can be represented in, given a tag of `tag_width` bits.
const fn min_width(n: u64, tag_width: TagWidth) -> EncodingWidth {
    assert_tag_width(tag_width);

    let max_inline = (1_u64 << tag_width) - 4;

    if n < max_inline {
        0
    } else if n < 256 {
        1
    } else if n < 256 * 256 {
        2
    } else if n < 256 * 256 * 256 * 256 {
        4
    } else {
        8
    }
}

const fn min_tag(n: u64, tag_width: TagWidth) -> Tag {
    let max_inline: u64 = (1_u64 << tag_width) - 4;

    let data = if n < max_inline {
        n as u8
    } else {
        let max_tag = maximal_tag(tag_width);

        if n < 256 {
            max_tag - 3
        } else if n < 256 * 256 {
            max_tag - 2
        } else if n < 256 * 256 * 256 * 256 {
            max_tag - 1
        } else {
            max_tag
        }
    };

    data
}

/// Returns the maximal tag of the given width as a Tag, i.e., `self.as_u8()` many one bits at the end, and everything else as zero bits. In other words, this computes `2^tag_width - 1`
const fn maximal_tag(tag_width: TagWidth) -> Tag {
    assert_tag_width(tag_width);

    ((1_u16 << tag_width) as u8).wrapping_sub(1)
}

/// An error indicating that a compact u64 encoding was not minimal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NotMinimal;

impl Display for NotMinimal {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "Expected a canonic compact u64 encoding, but got a non-minimal encoding instead.."
        )
    }
}

impl From<NotMinimal> for Blame {
    fn from(_value: NotMinimal) -> Self {
        Blame::TheirFault
    }
}

impl core::error::Error for NotMinimal {}

impl From<Infallible> for NotMinimal {
    fn from(_value: Infallible) -> Self {
        unreachable!()
    }
}

/// A wrapper around [`u64`], implementing the [ufotofu codec traits](codec).
///
/// The [encoding relation](codec) implemented by the [`Encodable`], [`EncodableKnownLength`], [`Decodable`], and [`DecodableCanonic`] impls of this type works by first encoding an eight-bit *tag* for the int, followed by the corresponding *int encoding*.
///
/// ```
/// use ufotofu::codec_prelude::*;
/// use compact_u64::*;
///
/// # pollster::block_on(async {
/// let mut buf = [0; 3];
/// let mut con = (&mut buf).into_consumer();
///
/// con.consume_encoded(&CompactU64(258)).await.unwrap();
/// assert_eq!(buf, [0xfd, 1, 2]);
///
/// let n: CompactU64 = producer::clone_from_slice(&buf[..]).produce_decoded().await.unwrap();
/// assert_eq!(n.0, 258);
/// # });
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct CompactU64(
    /// The wrapped [`u64`].
    pub u64,
);

impl From<u64> for CompactU64 {
    fn from(value: u64) -> Self {
        CompactU64(value)
    }
}

impl From<CompactU64> for u64 {
    fn from(value: CompactU64) -> Self {
        value.0
    }
}

#[cfg(feature = "dev")]
impl<'a> Arbitrary<'a> for CompactU64 {
    fn arbitrary(u: &mut arbitrary::Unstructured<'a>) -> arbitrary::Result<Self> {
        Ok(CompactU64(u64::arbitrary(u)?))
    }

    #[inline]
    fn size_hint(depth: usize) -> (usize, Option<usize>) {
        u64::size_hint(depth)
    }
}

/// Implements encoding by first encoding a byte storing the minimal 8-bit tag for self, followed by the corresponding compact u64 encoding.
impl Encodable for CompactU64 {
    /// Implements encoding by first encoding a byte storing the minimal 8-bit tag for self, followed by the corresponding compact u64 encoding.
    async fn encode<C>(&self, consumer: &mut C) -> Result<(), C::Error>
    where
        C: BulkConsumer<Item = u8> + ?Sized,
    {
        cu64_encode_standalone(self.0, consumer).await
    }
}

impl EncodableKnownLength for CompactU64 {
    fn len_of_encoding(&self) -> usize {
        1 + cu64_len_of_encoding(8, self.0)
    }
}

/// Implements decoding by first decoding a byte storing an 8-bit tag, followed by the corresponding compact u64 encoding.
impl Decodable for CompactU64 {
    type ErrorReason = Infallible;

    /// Implements decoding by first decoding a byte storing an 8-bit tag, followed by the corresponding compact u64 encoding.
    async fn decode<P>(
        producer: &mut P,
    ) -> Result<Self, DecodeError<P::Final, P::Error, Self::ErrorReason>>
    where
        P: BulkProducer<Item = u8> + ?Sized,
        Self: Sized,
    {
        Ok(Self(cu64_decode_standalone(producer).await?))
    }
}

/// Implements decoding by first decoding a byte storing an 8-bit tag, followed by the corresponding compact u64 encoding, and emitting an error if the encoding was not minimal.
impl DecodableCanonic for CompactU64 {
    type ErrorCanonic = NotMinimal;

    /// Implements decoding by first decoding a byte storing an 8-bit tag, followed by the corresponding compact u64 encoding, and emitting an error if the encoding was not minimal.
    async fn decode_canonic<P>(
        producer: &mut P,
    ) -> Result<Self, DecodeError<P::Final, P::Error, Self::ErrorCanonic>>
    where
        P: BulkProducer<Item = u8> + ?Sized,
        Self: Sized,
    {
        Ok(Self(cu64_decode_canonic_standalone(producer).await?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_min_width() {
        assert_eq!(0usize, min_width(11, 4));
        assert_eq!(1usize, min_width(12, 4));
        assert_eq!(1usize, min_width(255, 4));
        assert_eq!(2usize, min_width(256, 4));
        assert_eq!(2usize, min_width(65535, 4));
        assert_eq!(4usize, min_width(65536, 4));
        assert_eq!(4usize, min_width(4294967295, 4));
        assert_eq!(8usize, min_width(4294967296, 4));
        assert_eq!(8usize, min_width(18446744073709551615, 4));
    }

    #[test]
    fn test_maximal_tag() {
        assert_eq!(maximal_tag(5), 0b0001_1111);
    }
}
