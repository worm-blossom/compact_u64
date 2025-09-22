# Compact U64

A machine-friendly varint, implemented in Rust.

The general idea is the following:

- Each encoding is preceeded by a tag of two to eight (inclusive) bits.
- Each u64 can be encoded by setting the tag to the greatest possible number and
  then encoding the u64 as an eight-byte big-endian integer.
- Each u64 that fits into four bytes can be encoded by setting the tag to the
  second-greatest possible number and then encoding the u64 as an four-byte
  big-endian integer.
- Each u64 that fits into two bytes can be encoded by setting the tag to the
  third-greatest possible number and then encoding the u64 as an two-byte
  big-endian integer.
- Each u64 that fits into one byte can be encoded by setting the tag to the
  fourth-greatest possible number and then encoding the u64 as an one-byte
  big-endian integer.
- If the tag has more than two bits, then each u64 that is less than the
  fourth-greatest tag can be encoded in the tag directly, followed by no further
  bytes.
