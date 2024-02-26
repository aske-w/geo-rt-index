//
// Created by aske on 2/26/24.
//

#ifndef GEO_RT_INDEX_FACTORY_HPP
#define GEO_RT_INDEX_FACTORY_HPP

#include <bits/unique_ptr.h>

template<typename T>
class Factory {
public:
	virtual std::unique_ptr<T> Build() = 0;
	virtual ~Factory() = default;
};

#endif // GEO_RT_INDEX_FACTORY_HPP
