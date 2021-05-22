#ifndef INTACT_UTILS_H
#define INTACT_UTILS_H

void stitch(const int& index, Point& point, int16_t* ptr_pcl,
    uint8_t* ptr_img_GL, uint8_t* ptr_img_CV)
{
    ptr_pcl[3 * index + 0] = point.m_xyz[0]; // x
    ptr_pcl[3 * index + 1] = point.m_xyz[1]; // y
    ptr_pcl[3 * index + 2] = point.m_xyz[2]; // z

    ptr_img_CV[4 * index + 0] = point.m_bgra[0]; // blue
    ptr_img_CV[4 * index + 1] = point.m_bgra[1]; // green
    ptr_img_CV[4 * index + 2] = point.m_bgra[2]; // red
    ptr_img_CV[4 * index + 3] = point.m_bgra[3]; // alpha

    ptr_img_GL[3 * index + 0] = point.m_rgb[0]; // blue
    ptr_img_GL[3 * index + 1] = point.m_rgb[1]; // green
    ptr_img_GL[3 * index + 2] = point.m_rgb[2]; // red
}

#endif /*INTACT_UTILS_H*/
