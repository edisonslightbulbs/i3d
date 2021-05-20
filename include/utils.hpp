#ifndef INTACT_UTILS_H
#define INTACT_UTILS_H

void chromaPixel(const int& index, uint8_t* ptr_data)
{
    ptr_data[4 * index + 0] = 93;  // blue
    ptr_data[4 * index + 1] = 171; // green
    ptr_data[4 * index + 2] = 65;  // red
    ptr_data[4 * index + 3] = 0;   // alpha
}

bool vacant(
        const int& index, const short* ptr_data, const Point& min, const Point& max)
{
    if (max.m_xyz[2] == __FLT_MAX__ || min.m_xyz[2] == __FLT_MIN__) {
        return false;
    }
    if ((float)ptr_data[3 * index + 0] > max.m_xyz[0]
        || (float)ptr_data[3 * index + 0] < min.m_xyz[0]
        || (float)ptr_data[3 * index + 1] > max.m_xyz[1]
        || (float)ptr_data[3 * index + 1] < min.m_xyz[1]
        || (float)ptr_data[3 * index + 2] > max.m_xyz[2]
        || (float)ptr_data[3 * index + 2] < min.m_xyz[2]) {
        return false;
    }
    return true;
}
#endif /*INTACT_UTILS_H*/
