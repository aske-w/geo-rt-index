/**
 * Based on pseudo code from https://en.wikipedia.org/w/index.php?title=Quadtree&oldid=1188233918
 */

#include <cassert>
#include <memory>
#include <random>
#include <vector>
#include <cuda_runtime.h>
#include <thread>

#include "helpers/cuda_buffer.hpp"

namespace qt_gpu
{

using std::make_unique;
using std::unique_ptr;
using std::vector;

class Point
{
private:
	int32_t x, y;
public:
	Point() : x(0), y(0) { }
	Point(const float, const float) = delete;
	Point(const double, const double) = delete;
	Point(const int32_t _x, const int32_t _y) : x(_x), y(_y)
	{ }
	Point(const Point& other) : x(other.GetX()), y(other.GetY())
	{}
	Point& operator=(const Point& other) = default;
	inline int32_t GetX() const
	{
		return x;
	}
	inline int32_t GetY() const
	{
		return y;
	}
};

class BBox
{
public:
	const Point center;
	const int32_t half_dimension;
private:
	//! Lower left, upper right.
	Point ll, ur;
public:

	explicit BBox(const Point& _c, const int32_t _half_dimension)
	    : center(_c),
	      half_dimension(_half_dimension),
	      ll(Point(_c.GetX() - half_dimension, _c.GetY()- half_dimension)),
	      ur(Point(_c.GetX() + half_dimension, _c.GetY()+ half_dimension))
	{
		assert(_half_dimension > 0);
	}

	explicit BBox(const Point&& _c, const int32_t _half_dimension)
	    : center(std::move(_c)),
	      half_dimension(_half_dimension),
	      ll(Point(_c.GetX() - half_dimension, _c.GetY()- half_dimension)),
	      ur(Point(_c.GetX() + half_dimension, _c.GetY()+ half_dimension))
	{
		assert(_half_dimension > 0);
	}

	BBox(BBox&& other)
	    : center(other.center),
	      half_dimension(other.half_dimension),
	      ll(Point(other.center.GetX() - half_dimension, other.center.GetY()- half_dimension)),
	      ur(Point(other.center.GetX() + half_dimension, other.center.GetY()+ half_dimension))
	{
		assert(other.half_dimension > 0);
	}

	BBox(const BBox& other) : center(other.center),
	      half_dimension(other.half_dimension),
	      ll(Point{other.center.GetX() - half_dimension, other.center.GetY()- half_dimension}),
	      ur(Point(other.center.GetX() + half_dimension, other.center.GetY()+ half_dimension))
	{
		assert(other.half_dimension > 0);
	}

	BBox& operator=(const BBox& other)
	{
		memcpy(this, &other, sizeof(BBox));
		return *this;
	}

	inline bool contains(const Point& point) const
	{
		return ll.GetX() <= point.GetX() && point.GetX() <= ur.GetX() &&
			ll.GetY() <= point.GetY() && point.GetY() <= ur.GetY();
	}

	inline bool intersects(const BBox& other) const
	{	// from https://noonat.github.io/intersect/#aabb-vs-aabb
		int32_t dx = other.center.GetX() - this->center.GetX();
		int32_t px = (other.half_dimension + this->half_dimension) - abs(dx);
		int32_t dy = other.center.GetY()- this->center.GetY();
		int32_t py = (other.half_dimension + this->half_dimension) - abs(dy);
		if (py < 0 && px < 0)
			return false;
		return true; // TODO optimize
	}
};

template<const uint8_t NODE_CAPACITY = 4>
class QuadTree
{
private:
	uint32_t size;
	Point* points;
	Point* points_d;
//	std::vector<Point> points; // std::array?
	QuadTree* nw;
	QuadTree* ne;
	QuadTree* sw;
	QuadTree* se;

public:
	const BBox boundary;

	explicit QuadTree(const BBox&& _boundary)
	    : boundary(std::move(_boundary)),
	      points(new Point[NODE_CAPACITY]),
	      nw(nullptr),
	      ne(nullptr),
	      sw(nullptr),
	      se(nullptr), size(0)
	{
	}
	__host__ CUdeviceptr upload()
	{
		const auto count = NODE_CAPACITY * sizeof(Point);
		cudaMalloc(&points_d, count);
		cudaMemcpy(points_d, points, count, cudaMemcpyHostToDevice);
	}
	bool insert(const Point& p)
	{
//		if(!boundary.contains(p))
//			return false;
//
//		size++;
//
//		if(points && points->size() < NODE_CAPACITY && !nw)
//		{
//			points->push_back(std::move(p));
//			return true;
//		}
//
//		if (!nw) subdivide();
//
//		if(nw->insert(p)) return true;
//		if(ne->insert(p)) return true;
//		if(sw->insert(p)) return true;
//		if(se->insert(p)) return true;
//
//		return false; // TODO throw
	}

	void subdivide()
	{
//		assert(boundary.half_dimension > 1);
//		const auto quarter_dim = boundary.half_dimension >> 1;
//		const auto this_center = boundary.center;
//		const auto my_x = this_center.GetX();
//		const auto my_y = this_center.GetY();
//		nw = make_unique<QuadTree>(BBox{Point(my_x - quarter_dim, my_y + quarter_dim), quarter_dim});
//		ne = make_unique<QuadTree>(BBox{Point(my_x + quarter_dim, my_y + quarter_dim), quarter_dim});
//		se = make_unique<QuadTree>(BBox{Point(my_x + quarter_dim, my_y - quarter_dim), quarter_dim});
//		sw = make_unique<QuadTree>(BBox{Point(my_x - quarter_dim, my_y - quarter_dim), quarter_dim});
//
//		for (auto&& p : *points)
//		{
//			if (nw->boundary.contains(p))
//			{
//				nw->insert(p);
//				//continue;
//			}
//			else if (ne->boundary.contains(p))
//			{
//				ne->insert(p);
//				//continue;
//			}
//			else if (se->boundary.contains(p))
//			{
//				se->insert(p);
//				//continue;
//			}
//			else if (sw->boundary.contains(p))
//			{
//				sw->insert(p);
//				//continue;
//			}
//			else
//			{
//				throw std::exception();
//			}
//		}
//		this->points.reset();
	}

	__device__ void query_range(const BBox* range, Point* results, uint32_t* results_index)
	{
		atomicAdd(results_index, 1);
		printf("%d\n",*results_index);
		return;
//		if(!node->boundary.intersects(range))
//		{
//			return;
//		}
//
//		if(!node->nw)
//		{
////			results.insert(results.end(), points->begin(), points->end());
//			for(auto&& p : node->points)
//			{
//				if (range->contains(p))
//				{
//					results.push_back(p);
//				}
//			}
//		}
//		else
//		{
//			auto nw_result = node->nw->query_range(range);
//			results.insert(results.end(), nw_result.begin(), nw_result.end());
//
//			auto ne_result = node->ne->query_range(range);
//			results.insert(results.end(), ne_result.begin(), ne_result.end());
//
//			auto se_result = node->se->query_range(range);
//			results.insert(results.end(), se_result.begin(), se_result.end());
//
//			auto sw_result = node->sw->query_range(range);
//			results.insert(results.end(), sw_result.begin(), sw_result.end());
//		}
//
//		return;
	}
};

__global__ void query_range(qt_gpu::QuadTree<32>* node, const BBox* range, Point* results)
{
	printf("query_range\n");
	__shared__ uint32_t index;
	index = 0;
	node->query_range(range, results, &index);
}

template<const uint8_t NODE_CAPACITY>
void tests()
{
	BBox query{BBox(Point {2, 2}, 2)};
	{
		QuadTree<NODE_CAPACITY> t{BBox(Point {2, 2}, 2)};
		t.insert(Point(0, 0));
		t.insert(Point(4, 0));
		t.insert(Point(0, 4));
		t.insert(Point(4, 4));
		auto result = t.query_range(query);
		assert(result.size() == 4);
	}

	{
		QuadTree<NODE_CAPACITY> t {BBox(Point {4, 4}, 4)};
		t.insert(Point(0,0));
		t.insert(Point(2,2));
		auto result = t.query_range(query);
		assert(result.size() == 2);
	}

	{
		QuadTree<NODE_CAPACITY> t {BBox(Point {2, 2}, 2)};
		t.insert(Point(1,1));
		auto result = t.query_range(query);
		assert(result.size() == 1);
	}

//	{
//		QuadTree<NODE_CAPACITY> t {BBox(Point {1, 1}, 1)};
//		t.insert(Point(1,1));
//		auto result = t.query_range(query);
//		assert(result.size() == 1);
//	}
}

void tests_bbox_intersect()
{
	const int32_t factor = 4;
	// 1:1 overlap
	Point center{1 * factor, 1 * factor};
	int32_t half_dimension = 1 * factor;
	{
		BBox a{center, half_dimension};
		BBox b{center, half_dimension};
		assert(a.intersects(a));
		assert(a.intersects(b));
		assert(b.intersects(a));
		auto a_identity = a;
		a.intersects(a_identity);
	}

	auto a_center = center;
	// edges touch
	{
		BBox a{a_center, half_dimension};
		{	// top of a & bottom of b
			Point b_center{a_center.GetX(), a_center.GetY()+ half_dimension * 2};
			BBox b{b_center, half_dimension};
			assert(a.intersects(b));
			assert(b.intersects(a));
		}
		{	// bottom of a & top of b
			Point b_center{a_center.GetX(), a_center.GetY()- half_dimension * 2};
			BBox b{b_center, half_dimension};
			assert(a.intersects(b));
			assert(b.intersects(a));
		}
		{	// right of a & left of b
			Point b_center{a_center.GetX() + half_dimension * 2, a_center.GetY()};
			BBox b{b_center, half_dimension};
			assert(a.intersects(b));
			assert(b.intersects(a));
		}
		{	// right of b & left of a
			Point b_center{a_center.GetX() - half_dimension * 2, a_center.GetY()};
			BBox b{b_center, half_dimension};
			assert(a.intersects(b));
			assert(b.intersects(a));
		}
	}

	// no overlap
	{

	}

	// corners touch
	{
		BBox a{a_center, half_dimension};

		{	// tl a & br b
			Point b_center{a_center.GetX() - half_dimension * 2, a_center.GetY()+ half_dimension * 2};
			BBox b{b_center, half_dimension};
 			assert(a.intersects(b));
		}
		{	// bl a & tr b
			Point b_center{a_center.GetX() - half_dimension * 2, a_center.GetY()- half_dimension * 2};
			BBox b{b_center, half_dimension};
			assert(a.intersects(b));
		}
		{	// br a & tl b
			Point b_center{a_center.GetX() + half_dimension * 2, a_center.GetY()- half_dimension * 2};
			BBox b{b_center, half_dimension};
			assert(a.intersects(b));
		}
		{	// tr a & bl b
			Point b_center{a_center.GetX() + half_dimension * 2, a_center.GetY()+ half_dimension * 2};
			BBox b{b_center, half_dimension};
			assert(a.intersects(b));
		}
	}
}

void tests_bbox_contains()
{
	{	// on side, inside and in corners
		BBox b{Point(1,1), 1};
		for(int32_t i = 0; i < 3; i++)
		{
			for (int32_t j = 0; j < 3; j++)
			{
				Point p{i, j};
				assert(b.contains(p));
			}
		}

		b = BBox{Point(7,4), 1};
		for(int32_t i = 6; i < 9; i++)
		{
			for (int32_t j = 3; j < 6; j++)
			{
				Point p{i, j};
				auto result = b.contains(p);
				assert(result);
			}
		}
	}

	{ 	// outside
		BBox b{Point(10,10), 3};
		for(int32_t i = 0; i < 7; i++)
		{
			for(int32_t j = 0; j < 20; j++)
			{
				assert(!b.contains(Point{i, j}));
				assert(!b.contains(Point{j, i}));
			}
		}
		for(int32_t i = 14; i < 20; i++)
		{
			for(int32_t j = 0; j < 20; j++)
			{
				assert(!b.contains(Point{i, j}));
				assert(!b.contains(Point{j, i}));
			}
		}
	}
}

__global__ void dim3_test()
{
	printf("hello\n");
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	printf("x: %d, y: %d\n", x ,y);
}

} // qt_gpu

using namespace qt_gpu;

int main()
{
//	tests<1>();
//	tests<2>();
//	tests<3>();
//	tests<4>();
//	tests_bbox_intersect();
//	tests_bbox_contains();

//	printf("q\n");
//	dim3_test<<<2,4>>>();

//	uint32_t index = 0;
//	uint32_t* result_d_ptr;
//	cudaMalloc(&result_d_ptr, sizeof(uint32_t));
//	cudaMemcpy(result_d_ptr, &index, sizeof(uint32_t), cudaMemcpyHostToDevice);
	QuadTree<32> q(BBox(Point(1,1), 1));
	q.upload();
//	geo_rt_index::helpers::cuda_buffer b;
//	b.alloc_and_upload<QuadTree<32>>({q});

	qt_gpu::query_range<<<1,1>>>(&q, nullptr, nullptr);
	cudaDeviceSynchronize();
	return 0;
}