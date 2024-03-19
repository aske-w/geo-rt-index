/**
 * Based on pseudo code from https://en.wikipedia.org/w/index.php?title=Quadtree&oldid=1188233918
 */

#include <cassert>
#include <memory>
#include <random>
#include <vector>

using std::make_unique;
using std::unique_ptr;
using std::vector;

class Point
{
private:
	int32_t x, y;
public:
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
	unique_ptr<std::vector<Point>> points; // std::array?
	unique_ptr<QuadTree> nw, ne, sw, se;

public:
	const BBox boundary;

	explicit QuadTree(const BBox& _boundary)
	    : boundary(std::move(_boundary)),
	      points(make_unique<vector<Point>>()),
	      nw(nullptr),
	      ne(nullptr),
	      sw(nullptr),
	      se(nullptr), size(0)
	{
	}

	bool insert(const Point& p)
	{
		if(!boundary.contains(p))
			return false;

		size++;

		if(points && points->size() < NODE_CAPACITY && !nw)
		{
			points->push_back(std::move(p));
			return true;
		}

		if (!nw) subdivide();

		if(nw->insert(p)) return true;
		if(ne->insert(p)) return true;
		if(sw->insert(p)) return true;
		if(se->insert(p)) return true;

		return false; // TODO throw
	}

	void subdivide()
	{
		assert(boundary.half_dimension > 1);
		const auto quarter_dim = boundary.half_dimension >> 1;
		const auto this_center = boundary.center;
		const auto my_x = this_center.GetX();
		const auto my_y = this_center.GetY();
		nw = make_unique<QuadTree>(BBox{Point(my_x - quarter_dim, my_y + quarter_dim), quarter_dim});
		ne = make_unique<QuadTree>(BBox{Point(my_x + quarter_dim, my_y + quarter_dim), quarter_dim});
		se = make_unique<QuadTree>(BBox{Point(my_x + quarter_dim, my_y - quarter_dim), quarter_dim});
		sw = make_unique<QuadTree>(BBox{Point(my_x - quarter_dim, my_y - quarter_dim), quarter_dim});

		for (auto&& p : *points)
		{
			if (nw->boundary.contains(p))
			{
				nw->insert(p);
				//continue;
			}
			else if (ne->boundary.contains(p))
			{
				ne->insert(p);
				//continue;
			}
			else if (se->boundary.contains(p))
			{
				se->insert(p);
				//continue;
			}
			else if (sw->boundary.contains(p))
			{
				sw->insert(p);
				//continue;
			}
			else
			{
				throw std::exception();
			}
		}
		this->points.reset();
	}

	vector<Point> query_range(const BBox& range) const
	{
		vector<Point> results;

		if(!boundary.intersects(range))
		{
			return results;
		}

		if(!nw)
		{
//			results.insert(results.end(), points->begin(), points->end());
			for(auto&& p : *points)
			{
				if (range.contains(p))
				{
					results.push_back(p);
				}
			}
		}
		else
		{
			auto nw_result = nw->query_range(range);
			results.insert(results.end(), nw_result.begin(), nw_result.end());

			auto ne_result = ne->query_range(range);
			results.insert(results.end(), ne_result.begin(), ne_result.end());

			auto se_result = se->query_range(range);
			results.insert(results.end(), se_result.begin(), se_result.end());

			auto sw_result = sw->query_range(range);
			results.insert(results.end(), sw_result.begin(), sw_result.end());
		}

		return results;
	}
};

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
	// outside
}

int main()
{
	tests<1>();
	tests<2>();
	tests<3>();
	tests<4>();
	tests_bbox_intersect();
	tests_bbox_contains();
//	QuadTree<1> t{BBox(Point{1, 1}, 1)};




//	std::random_device rd;
//	std::mt19937_64 gen {rd()};
//	std::uniform_real_distribution<int32_t> dis{0, 0.25};
//	for (uint8_t i = 0; i < 10; i++)
//	{
//		auto x = dis(gen);
//		auto y = dis(gen);
//		t.insert(Point(x,y));
//	}
	return 0;
}