/*
 * Diagonal.cpp
 *
 */

#include <OT/BitDiagonal.h>

void BitDiagonal::pack(octetStream& os) const
{
    for (int i = 0; i < N_ROWS; i++)
        os.store_bit(rows[i].get_bit(i));
    os.append(0);
}

void BitDiagonal::unpack(octetStream& os)
{
    *this = {};
    for (int i = 0; i < N_ROWS; i++)
        rows[i] = RowType(os.get_bit()) << i;
}
