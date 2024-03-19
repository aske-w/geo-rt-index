/**
 * Based on pseudo code from https://en.wikipedia.org/w/index.php?title=Quadtree&oldid=1188233918
 */

#include <vector>
#include <memory>
#include <random>

using std::vector;
using std::unique_ptr;
using std::make_unique;

class Point
{
public:
	const float x, y;
	Point(const float _x, const float _y) : x(_x), y(_y)
	{
	}
//	Point(Point* p) : Point(*p)
//	{}
	Point(const Point& other) : x(other.x), y(other.y)
	{}
	Point(Point&& other) : x(other.x), y(other.y)
	{}
	Point& operator=(const Point& other)
	{
		return *this = Point(other);
	}
	Point& operator=(Point&& other)
	{
		return other;
	}
//	explicit Point(Point&& other) : x(other.x), y(other.y)
//	{}
};

class BBox
{
public:
	const Point center;
	const float half_dimension;
private:
	//! Lower left, upper right.
	const Point ll, ur;
public:

	explicit BBox(const Point&& _c, const float _half_dimension)
	    : center(std::move(_c)),
	      half_dimension(_half_dimension),
	      ll(Point(_c.x - half_dimension, _c.y - half_dimension)),
	      ur(Point(_c.x + half_dimension, _c.y + half_dimension))
	{
	}

	BBox(BBox&& other)
	    : center(other.center),
	      half_dimension(other.half_dimension),
	      ll(Point(other.center.x - half_dimension, other.center.y - half_dimension)),
	      ur(Point(other.center.x + half_dimension, other.center.y + half_dimension))
	{}

	BBox(const BBox& other) : center(other.center),
	      half_dimension(other.half_dimension),
	      ll(Point(other.center.x - half_dimension, other.center.y - half_dimension)),
	      ur(Point(other.center.x + half_dimension, other.center.y + half_dimension))
	{}

	inline bool contains(const Point& point) const
	{
		return ll.x <= point.x && point.x < ur.x &&
			ll.y <= point.y && point.y < ur.y;
	}

	inline bool intersects(const BBox& other) const
	{
		return !(ur.x < other.ll.x ||
		       ll.x < other.ur.x ||
		       ur.y < other.ll.y ||
		       ll.y < other.ur.y);
	}
};

template<const uint8_t NODE_CAPACITY = 4>
class QuadTree
{
private:
	uint32_t size;
	unique_ptr<vector<Point>> points; // std::array?
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
		const auto quarter_dim = boundary.half_dimension / 2;
		const auto this_center = boundary.center;
		const auto my_x = this_center.x;
		const auto my_y = this_center.y;
		nw = make_unique<QuadTree>(BBox {Point(my_x - quarter_dim, my_y + quarter_dim), quarter_dim});
		ne = make_unique<QuadTree>(BBox {Point(my_x + quarter_dim, my_y + quarter_dim), quarter_dim});
		se = make_unique<QuadTree>(BBox {Point(my_x + quarter_dim, my_y - quarter_dim), quarter_dim});
		sw = make_unique<QuadTree>(BBox {Point(my_x - quarter_dim, my_y - quarter_dim), quarter_dim});

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
				std::exception();
			}
		}
		this->points.reset();
	}

	void query_range() const
	{
	}
};

int main()
{
	QuadTree<4> t{BBox(Point{0.5f, 0.5f}, 0.5f)};
	std::random_device rd;
	std::mt19937_64 gen {rd()};
	std::uniform_real_distribution<float> dis{0, 0.25};

	//t.insert(Point(0.25,0.25));
	/*t.insert(Point(0.75,0.25));
	t.insert(Point(0.25,0.75));
	t.insert(Point(0.75,0.75));
	t.insert(Point(0.70,0.75));*/


	for (uint8_t i = 0; i < 10; i++)
	{
		auto x = dis(gen);
		auto y = dis(gen);
		t.insert(Point(x,y));
	}
	return 0;
}