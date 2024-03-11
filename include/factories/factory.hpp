//
// Created by aske on 2/26/24.
//

#ifndef GEO_RT_INDEX_FACTORY_HPP
#define GEO_RT_INDEX_FACTORY_HPP

#include <bits/unique_ptr.h>

namespace geo_rt_index
{
namespace factories
{

template<typename BUILD_TYPE>
class Factory {
public:
	virtual std::unique_ptr<BUILD_TYPE> Build() = 0;
	virtual ~Factory() = default;
};

} // factories
} // geo_rt_index


#endif // GEO_RT_INDEX_FACTORY_HPP
