#![no_main]

use compact_u64::*;
use libfuzzer_sys::fuzz_target;
use ufotofu::codec_prelude::*;
use ufotofu::producer::clone_from_slice;

fuzz_target!(|data: (u8, u8, u64)| {
    let (tag_width, tag_offset, n) = data;

    if tag_width.saturating_add(tag_offset) <= 8 && tag_width >= 2 {
        pollster::block_on(async {
            let mut buf = vec![99u8; 1 + cu64_len_of_encoding(tag_width, n)];
            let mut con = (&mut buf).into_consumer();

            let mut tag = 0;
            write_tag(&mut tag, tag_width, tag_offset, n);

            con.consume_item(tag).await.unwrap();
            cu64_encode(n, tag_width, &mut con).await.unwrap();

            let mut pro = clone_from_slice(&buf[..]);
            let decoded_tag = pro.produce_item().await.unwrap();
            assert_eq!(decoded_tag, tag);

            let decoded_u64 = cu64_decode_canonic(decoded_tag, tag_width, tag_offset, &mut pro)
                .await
                .unwrap();
            assert_eq!(decoded_u64, n);
        });
    }
});
