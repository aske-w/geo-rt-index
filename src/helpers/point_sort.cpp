//
// Created by aske on 5/9/24.
//

#include "helpers/point_sort.hpp"
#include <stdexcept>
#include <execution>
#include <algorithm>

namespace geo_rt_index
{
namespace helpers
{

using types::Point;
using types::PointSorting;

//! from https://stackoverflow.com/a/26856331
static uint64_t mortonIndex(const Point& p)
{
	// Pun the x and y coordinates as integers: Just re-interpret the bits.
	//
	auto ix = reinterpret_cast<const uint32_t &>(p.x);
	auto iy = reinterpret_cast<const uint32_t &>(p.y);

	// Since we're assuming 2s complement arithmetic (99.99% of hardware today),
	// we'll need to convert these raw integer-punned floats into
	// their corresponding integer "indices".

	// Smear their sign bits into these for twiddling below.
	//
	const auto ixs = static_cast<int32_t>(ix) >> 31;
	const auto iys = static_cast<int32_t>(iy) >> 31;

	// This is a combination of a fast absolute value and a bias.
	//
	// We need to adjust the values so -FLT_MAX is close to 0.
	//
	ix = (((ix & 0x7FFFFFFFL) ^ ixs) - ixs) + 0x7FFFFFFFL;
	iy = (((iy & 0x7FFFFFFFL) ^ iys) - iys) + 0x7FFFFFFFL;

	// Now we have -FLT_MAX close to 0, and FLT_MAX close to UINT_MAX,
	// with everything else in-between.
	//
	// To make this easy, we'll work with x and y as 64-bit integers.
	//
	uint64_t xx = ix;
	uint64_t yy = iy;

	// Dilate and combine as usual...

	xx = (xx | (xx << 16)) & 0x0000ffff0000ffffLL;
	yy = (yy | (yy << 16)) & 0x0000ffff0000ffffLL;

	xx = (xx | (xx <<  8)) & 0x00ff00ff00ff00ffLL;
	yy = (yy | (yy <<  8)) & 0x00ff00ff00ff00ffLL;

	xx = (xx | (xx <<  4)) & 0x0f0f0f0f0f0f0f0fLL;
	yy = (yy | (yy <<  4)) & 0x0f0f0f0f0f0f0f0fLL;

	xx = (xx | (xx <<  2)) & 0x3333333333333333LL;
	yy = (yy | (yy <<  2)) & 0x3333333333333333LL;

	xx = (xx | (xx <<  1)) & 0x5555555555555555LL;
	yy = (yy | (yy <<  1)) & 0x5555555555555555LL;

	return xx | (yy << 1);
}

static bool CompareX(const Point& a, const Point&b)
{
	return a.x < b.x;
}

static bool CompareY(const Point& a, const Point&b)
{
	return a.y < b.y;
}

static bool CompareZOrder(const Point& a, const Point&b)
{
	auto z_a = mortonIndex(a);
	auto z_b = mortonIndex(b);
	return z_a < z_b;
}

void geo_rt_index::helpers::PointSort::Sort(std::vector<geo_rt_index::types::Point>& points, const geo_rt_index::types::PointSorting sortBy)
{
	static const auto policy = std::execution::par;
  	switch (sortBy)
	{
		case PointSorting::X:
		    std::sort(policy, points.begin(), points.end(), CompareX);
		    break;
		case PointSorting::Y:
		    std::sort(policy, points.begin(), points.end(), CompareY);
		    break;
		case PointSorting::ZOrderCurve:
		    std::sort(policy, points.begin(), points.end(), CompareZOrder);
		    break;
		default:
			throw std::runtime_error("");
	}
}


}
}
