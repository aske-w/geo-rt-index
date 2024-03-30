//
// Created by aske on 3/22/24.
//

#ifndef GEO_RT_INDEX_PRETTY_PRINTERS_HPP
#define GEO_RT_INDEX_PRETTY_PRINTERS_HPP

#include <vector>
#include <algorithm>
#include <iterator>
#include "types.hpp"

template<typename T>
std::ostream & operator<<(std::ostream & os, std::vector<T> vec)
{
	os<<"[";
	std::copy(vec.begin(), vec.end(), std::ostream_iterator<T>(os, "; "));
	os<<"]";
	return os;
}

std::ostream & operator<<(std::ostream & os, Point p)
{
	os << "(" << std::to_string(p.x) << ", " << std::to_string(p.y) << ")";
	return os;
}

#endif // GEO_RT_INDEX_PRETTY_PRINTERS_HPP
