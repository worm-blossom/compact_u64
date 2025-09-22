#![no_main]

use ufotofu_codec::fuzz_absolute_all;

fuzz_absolute_all!(compact_u64::CompactU64);
