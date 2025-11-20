#![no_main]

use codec::proptest::assert_codec_canonic_and_known_len;
use compact_u64::CompactU64;
use libfuzzer_sys::fuzz_target;
use ufotofu::codec_prelude::*;

fuzz_target!(|data: (
    CompactU64,
    CompactU64,
    TestConsumer<u8, (), ()>,
    TestConsumer<u8, (), ()>,
    TestProducer<u8, (), ()>,
    TestProducer<u8, (), ()>,
)| {
    let (t1, t2, c1, c2, p1, p2) = data;

    // The `pollster` crate lets you run async code in a sync closure.
    pollster::block_on(async {
        assert_codec_canonic_and_known_len(&t1, &t2, c1, c2, p1, p2).await;
    });
});
