#define _POSIX_SOURCE 1

#include "../pngpriv.h"

#include <stdint.h>

#include <smmintrin.h>

#ifndef PNG_ALIGNED_MEMORY_SUPPORTED
#  error "ALIGNED_MEMORY is required; set: -DPNG_ALIGNED_MEMORY_SUPPORTED"
#endif

void
png_read_filter_row_up_avx2(png_row_infop row_info, png_bytep row,
   png_const_bytep prev_row)
{
    png_size_t i;
    png_size_t istop = row_info->rowbytes;
    png_bytep rp = row;
    png_const_bytep pp = prev_row;

    for (i = 0; i < istop; i+=64, rp+=64, pp+=64) {
        for (int j = 0; j < 64; j+=16) {
            _mm_stream_si128((__m128i*)(rp+j),
                _mm_add_epi8(
                    _mm_load_si128((__m128i*)(rp+j)),
                    _mm_load_si128((__m128i*)(pp+j))
                ));
        }
    }
}

void
png_read_filter_row_sub4_avx2(png_row_infop row_info, png_bytep row,
   png_const_bytep prev_row)
{
    const unsigned int bpp = 4;
    png_bytep rp = row + bpp;
    png_const_bytep rp_end = row + row_info->rowbytes;

    PNG_UNUSED(prev_row)

    // Align rp to a 64 bytes boundary.
    while (((uintptr_t)rp) & 0x3F) {
        *rp = (png_byte)(((int)(*rp) + (int)(*(rp-bpp))) & 0xff);
        rp++;
    }

    __m128i left = _mm_load_si128((__m128i*)(rp-16));
    left = _mm_srli_si128(left, 12);
    while (rp < rp_end) {
        for (int i = 0; i < 64; i += 16) {
            // Load next data
            __m128i data = _mm_loadu_si128((__m128i*)(rp+i));

            // Creaet diff
            __m128i diff = _mm_slli_si128(data, 4);
            diff = _mm_or_si128(diff, left);

            data = _mm_add_epi8(data, diff);
            diff = _mm_slli_si128(diff, 4);
            data = _mm_add_epi8(data, diff);
            diff = _mm_slli_si128(diff, 4);
            data = _mm_add_epi8(data, diff);
            diff = _mm_slli_si128(diff, 4);
            data = _mm_add_epi8(data, diff);

            // Prepare next diff
            diff = _mm_srli_si128(data, 12);
            // Store data
            _mm_stream_si128((__m128i*)(rp+i), data);

            left = diff;
        }
        rp += 64;
    }
}

void
png_read_filter_row_avg4_avx2(png_row_infop row_info, png_bytep row,
   png_const_bytep prev_row)
{
   const unsigned int bpp = 4;
   png_size_t i;
   png_bytep rp = row;
   png_const_bytep pp = prev_row;
   png_size_t istop = row_info->rowbytes - bpp;

   for (i = 0; i < bpp; i++) {
      *rp = (png_byte)(((int)(*rp) +
         ((int)(*pp++) / 2 )) & 0xff);

      rp++;
   }

   for (i = 0; i < istop; i++) {
      *rp = (png_byte)(((int)(*rp) +
         (int)(*pp++ + *(rp-bpp)) / 2 ) & 0xff);

      rp++;
   }
}

void
png_read_filter_row_paeth4_avx2(png_row_infop row_info, png_bytep row,
   png_const_bytep prev_row)
{
    const unsigned int bpp = 4;
    png_bytep rp_end = row + bpp;

    /* Process the first pixel in the row completely (this is the same as 'up'
    * because there is only one candidate predictor for the first row).
    */
    while (row < rp_end) {
        int a = *row + *prev_row++;
        *row++ = (png_byte)a;
    }

    /* Remainder */
    rp_end += row_info->rowbytes - bpp;

    const __m128i selector0 = _mm_setr_epi8(
        0x00, 0x02, 0x04, 0x06,
        0xFF, 0xFF, 0xFF, 0xFF,
        0xFF, 0xFF, 0xFF, 0xFF,
        0xFF, 0xFF, 0xFF, 0xFF
    );
    const __m128i selector1 = _mm_setr_epi8(
        0xFF, 0xFF, 0xFF, 0xFF,
        0x08, 0x0A, 0x0C, 0x0E,
        0xFF, 0xFF, 0xFF, 0xFF,
        0xFF, 0xFF, 0xFF, 0xFF
    );

    while (row < rp_end) {
        __m128i xc = _mm_cvtepu8_epi16(_mm_loadu_si128((__m128i*)(prev_row - bpp)));
        __m128i xa = _mm_cvtepu8_epi16(_mm_loadu_si128((__m128i*)(row - bpp)));
        __m128i xb = _mm_cvtepu8_epi16(_mm_loadu_si128((__m128i*)(prev_row)));

        __m128i xrow = _mm_loadu_si128((__m128i*)(row));

        // First four bytes
        {
            __m128i xp = _mm_subs_epi16(xb, xc);
            __m128i xpc = _mm_subs_epi16(xa, xc);

            __m128i xpa = _mm_abs_epi16(xp);
            __m128i xpb = _mm_abs_epi16(xpc);
            xpc = _mm_abs_epi16(_mm_add_epi16(xp, xpc));

            __m128i mask = _mm_cmplt_epi16(xpb, xpa);
            xpa = _mm_min_epi16(xpb, xpa);
            __m128i xa_ = _mm_blendv_epi8(xa, xb, mask);

            mask = _mm_cmplt_epi16(xpc, xpa);
            xa_ = _mm_blendv_epi8(xa_, xc, mask);

            __m128i diff = _mm_shuffle_epi8(xa_, selector0);

            xa_ = _mm_slli_si128(xa_, 8);
            xa = _mm_add_epi8(xa, xa_);

            xrow = _mm_add_epi8(xrow, diff);
        }


        // Second four bytes
        {
            __m128i xp = _mm_subs_epi16(xb, xc);
            __m128i xpc = _mm_subs_epi16(xa, xc);

            __m128i xpa = _mm_abs_epi16(xp);
            __m128i xpb = _mm_abs_epi16(xpc);
            xpc = _mm_abs_epi16(_mm_add_epi16(xp, xpc));

            __m128i mask = _mm_cmplt_epi16(xpb, xpa);
            xpa = _mm_min_epi16(xpb, xpa);
            __m128i xa_ = _mm_blendv_epi8(xa, xb, mask);

            mask = _mm_cmplt_epi16(xpc, xpa);
            xa_ = _mm_blendv_epi8(xa_, xc, mask);

            __m128i diff = _mm_shuffle_epi8(xa_, selector1);

            xrow = _mm_add_epi8(xrow, diff);
        }

        _mm_storeu_si128((__m128i*)row, xrow);

        prev_row += 8;
        row += 8;
    }
}

void
png_read_filter_row_sub3_avx2(png_row_infop row_info, png_bytep row,
   png_const_bytep prev_row)
{
   png_size_t i;
   png_size_t istop = row_info->rowbytes;
   const unsigned int bpp = 3;
   png_bytep rp = row + bpp;

   PNG_UNUSED(prev_row)

   for (i = bpp; i < istop; i++) {
      *rp = (png_byte)(((int)(*rp) + (int)(*(rp-bpp))) & 0xff);
      rp++;
   }
}

void
png_read_filter_row_avg3_avx2(png_row_infop row_info, png_bytep row,
   png_const_bytep prev_row)
{
   const unsigned int bpp = 3;
   png_size_t i;
   png_bytep rp = row;
   png_const_bytep pp = prev_row;
   png_size_t istop = row_info->rowbytes - bpp;

   for (i = 0; i < bpp; i++) {
      *rp = (png_byte)(((int)(*rp) +
         ((int)(*pp++) / 2 )) & 0xff);

      rp++;
   }

   for (i = 0; i < istop; i++) {
      *rp = (png_byte)(((int)(*rp) +
         (int)(*pp++ + *(rp-bpp)) / 2 ) & 0xff);

      rp++;
   }
}

void
png_read_filter_row_paeth3_avx2(png_row_infop row_info, png_bytep row,
   png_const_bytep prev_row)
{
    const unsigned int bpp = 3;
    png_bytep rp_end = row + bpp;

    /* Process the first pixel in the row completely (this is the same as 'up'
    * because there is only one candidate predictor for the first row).
    */
    while (row < rp_end) {
        int a = *row + *prev_row++;
        *row++ = (png_byte)a;
    }

    /* Remainder */
    rp_end += row_info->rowbytes - bpp;

    const __m128i selector0 = _mm_setr_epi8(
        0x00, 0x02, 0x04, 0xFF,
        0xFF, 0xFF, 0xFF, 0xFF,
        0xFF, 0xFF, 0xFF, 0xFF,
        0xFF, 0xFF, 0xFF, 0xFF
    );
    const __m128i selector1 = _mm_setr_epi8(
        0xFF, 0xFF, 0xFF, 0x06,
        0x08, 0x0A, 0xFF, 0xFF,
        0xFF, 0xFF, 0xFF, 0xFF,
        0xFF, 0xFF, 0xFF, 0xFF
    );

    while (row < rp_end) {
        __m128i xc = _mm_cvtepu8_epi16(_mm_loadu_si128((__m128i*)(prev_row - bpp)));
        __m128i xa = _mm_cvtepu8_epi16(_mm_loadu_si128((__m128i*)(row - bpp)));
        __m128i xb = _mm_cvtepu8_epi16(_mm_loadu_si128((__m128i*)(prev_row)));

        __m128i xrow = _mm_loadu_si128((__m128i*)(row));

        // First three bytes
        {
            __m128i xp = _mm_subs_epi16(xb, xc);
            __m128i xpc = _mm_subs_epi16(xa, xc);

            __m128i xpa = _mm_abs_epi16(xp);
            __m128i xpb = _mm_abs_epi16(xpc);
            xpc = _mm_abs_epi16(_mm_add_epi16(xp, xpc));

            __m128i mask = _mm_cmplt_epi16(xpb, xpa);
            xpa = _mm_min_epi16(xpb, xpa);
            __m128i xa_ = _mm_blendv_epi8(xa, xb, mask);

            mask = _mm_cmplt_epi16(xpc, xpa);
            xa_ = _mm_blendv_epi8(xa_, xc, mask);

            __m128i diff = _mm_shuffle_epi8(xa_, selector0);

            xa_ = _mm_slli_si128(xa_, 10);
            xa_ = _mm_srli_si128(xa_, 4);
            xa = _mm_add_epi8(xa, xa_);

            xrow = _mm_add_epi8(xrow, diff);
        }


        // Second three bytes
        {
            __m128i xp = _mm_subs_epi16(xb, xc);
            __m128i xpc = _mm_subs_epi16(xa, xc);

            __m128i xpa = _mm_abs_epi16(xp);
            __m128i xpb = _mm_abs_epi16(xpc);
            xpc = _mm_abs_epi16(_mm_add_epi16(xp, xpc));

            __m128i mask = _mm_cmplt_epi16(xpb, xpa);
            xpa = _mm_min_epi16(xpb, xpa);
            __m128i xa_ = _mm_blendv_epi8(xa, xb, mask);

            mask = _mm_cmplt_epi16(xpc, xpa);
            xa_ = _mm_blendv_epi8(xa_, xc, mask);

            __m128i diff = _mm_shuffle_epi8(xa_, selector1);

            xrow = _mm_add_epi8(xrow, diff);
        }

        _mm_storeu_si128((__m128i*)row, xrow);

        prev_row += 6;
        row += 6;
    }
}

