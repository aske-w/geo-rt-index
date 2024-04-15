//
// Created by aske on 4/14/24.
//

#ifndef GEO_RT_INDEX_POINT_HPP
#define GEO_RT_INDEX_POINT_HPP

namespace geo_rt_index
{
namespace types
{

struct Point {
	float x, y;

	Point(int _x, int _y) : x(static_cast<float>(_x)), y(static_cast<float>(_y))
	{
	}
	Point(float _x, float _y) : x(_x), y(_y)
	{
	}
};

}
}


#endif // GEO_RT_INDEX_POINT_HPP
