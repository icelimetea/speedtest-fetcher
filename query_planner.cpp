#include <cstdint>
#include <cmath>
#include <numbers>
#include <limits>
#include <vector>
#include <algorithm>
#include <charconv>
#include <random>
#include <fstream>

#include <absl/container/flat_hash_set.h>

#include <simdjson.h>

#include <CGAL/Unique_hash_map.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Delaunay_triangulation_on_sphere_traits_2.h>
#include <CGAL/Delaunay_triangulation_on_sphere_2.h>

using LinearKernel = CGAL::Exact_predicates_inexact_constructions_kernel;
using SphericalDelaunayTraits = CGAL::Delaunay_triangulation_on_sphere_traits_2<LinearKernel>;
using SphericalDelaunay = CGAL::Delaunay_triangulation_on_sphere_2<SphericalDelaunayTraits>;

using Point3 = SphericalDelaunay::Point_3;

using ServerID = int32_t;

constexpr double EARTH_RADIUS_MILES = 3963.1676;

constexpr size_t SHORT_RANGE_SERVERS = 100;
constexpr size_t LONG_RANGE_SERVERS = 20;
constexpr size_t MAX_SERVERS = std::max(SHORT_RANGE_SERVERS, LONG_RANGE_SERVERS);

constexpr double SHORT_SEARCH_RANGE = 30.0;
constexpr double LONG_SEARCH_RANGE = 2000.0;

const double SHORT_RANGE_COSINE = std::cos(SHORT_SEARCH_RANGE / EARTH_RADIUS_MILES);
const double LONG_RANGE_COSINE = std::cos(LONG_SEARCH_RANGE / EARTH_RADIUS_MILES);

class GeographicPoint {
private:
	double lat;
	double lon;
public:
	GeographicPoint(std::string_view latitude, std::string_view longtitude) {
		if (std::from_chars(latitude.data(), latitude.data() + latitude.size(), this->lat).ec != std::errc())
			throw std::invalid_argument("Invalid latitude");

		if (std::from_chars(longtitude.data(), longtitude.data() + longtitude.size(), this->lon).ec != std::errc())
			throw std::invalid_argument("Invalid longtitude");
	}

	GeographicPoint(const Point3& point) {
		this->lat = std::asin(point.z()) * 180 / std::numbers::pi;
		this->lon = std::atan2(point.y(), point.x()) * 180 / std::numbers::pi;
	}

	double getLatitude() const {
		return this->lat;
	}

	double getLongtitude() const {
		return this->lon;
	}

	Point3 toPoint() const {
		double latRad = this->lat * std::numbers::pi / 180;
		double lonRad = this->lon * std::numbers::pi / 180;

		double x = std::cos(latRad) * std::cos(lonRad);
		double y = std::cos(latRad) * std::sin(lonRad);
		double z = std::sin(latRad);

		return Point3(x, y, z);
	}
};

class RandomPointGenerator {
private:
	using RandomGenerator = std::mt19937;

	RandomGenerator rng;
	std::normal_distribution<> dist;
public:
	RandomPointGenerator() {
		std::random_device seedRng;
		this->rng = RandomGenerator(seedRng());
	}

	Point3 operator()() {
		double x = this->dist(this->rng);
		double y = this->dist(this->rng);
		double z = this->dist(this->rng);

		double length = std::sqrt(x * x + y * y + z * z);

		return Point3(x, y, z, length);
	}
};

class Queries {
private:
	using ServerIterator = std::vector<ServerID>::const_iterator;

	struct QueryTag {
		Point3 location;
		size_t endIndex;
	};

	std::vector<ServerID> servers;
	std::vector<QueryTag> queries;
public:
	class Query {
	private:
		const Point3& location;
		ServerIterator serversBegin;
		ServerIterator serversEnd;
	public:
		Query(const Point3& location, ServerIterator begin, ServerIterator end) :
			location(location),
			serversBegin(begin),
			serversEnd(end) {}

		const Point3& getLocation() const {
			return this->location;
		}

		size_t size() const {
			return this->serversEnd - this->serversBegin;
		}

		ServerIterator begin() const {
			return this->serversBegin;
		}

		ServerIterator end() const {
			return this->serversEnd;
		}
	};

	Queries(size_t queryCount) {
		this->servers.reserve(MAX_SERVERS * queryCount);
		this->queries.reserve(queryCount);
		this->endQuery(Point3());
	}

	Queries(const Queries& other) = delete;

	Queries(Queries&& other) noexcept :
		servers(std::move(other.servers)),
		queries(std::move(other.queries)) {}

	Queries& operator=(const Queries& other) = delete;

	Queries& operator=(Queries&& other) noexcept {
		this->servers = std::move(other.servers);
		this->queries = std::move(other.queries);
		return *this;
	}

	void endQuery(const Point3& location) {
		this->queries.emplace_back(location, this->servers.size());
	}

	void insertServer(ServerID serverID) {
		this->servers.push_back(serverID);
	}

	size_t bufferedServers() const {
		return this->servers.size() - this->queries.back().endIndex;
	}

	size_t size() const {
		return this->queries.size() - 1;
	}

	Query operator[](size_t index) const {
		size_t serversStart = this->queries[index].endIndex;
		size_t serversEnd = this->queries[index + 1].endIndex;
		return Query(this->queries[index + 1].location, this->servers.cbegin() + serversStart, this->servers.cbegin() + serversEnd);
	}
};

class QueryBuilder {
private:
	using VertexSet = absl::flat_hash_set<SphericalDelaunay::Vertex_handle>;
	using VertexMap = CGAL::Unique_hash_map<SphericalDelaunay::Vertex_handle, std::vector<ServerID>>;

	SphericalDelaunay delaunay;
	VertexMap buckets;

	struct Neighbour {
		SphericalDelaunay::Vertex_handle vertex;
		SphericalDelaunay::FT distance;

		Neighbour(const Point3& origin, const SphericalDelaunay& delaunay, SphericalDelaunay::Vertex_handle vertexHandle) :
			vertex(vertexHandle) {
			const Point3& location = delaunay.point(vertexHandle);
			this->distance = origin.x() * location.x() + origin.y() * location.y() + origin.z() * location.z();
		}

		bool operator<(const Neighbour& other) const {
			return this->distance < other.distance;
		}
	};

	void dijkstraSearch(const Point3& origin,
			    std::vector<Neighbour>& vertices,
			    VertexSet& reached,
			    Queries& queries) const {
		while (!vertices.empty()) {
			const Neighbour& neighbour = vertices.front();

			size_t limit;

			if (neighbour.distance > SHORT_RANGE_COSINE) {
				limit = SHORT_RANGE_SERVERS;
			} else if (neighbour.distance > LONG_RANGE_COSINE) {
				limit = LONG_RANGE_SERVERS;
			} else {
				queries.endQuery(origin);
				return;
			}

			for (ServerID serverID : this->buckets[neighbour.vertex]) {
				if (queries.bufferedServers() >= limit) {
					queries.endQuery(origin);
					return;
				}

				queries.insertServer(serverID);
			}

			auto incidents = this->delaunay.incident_vertices(neighbour.vertex);
			auto nextVertex = incidents;

			std::pop_heap(vertices.begin(), vertices.end());
			vertices.pop_back();

			do {
				if (reached.insert(nextVertex).second) {
					vertices.emplace_back(origin, this->delaunay, nextVertex);
					std::push_heap(vertices.begin(), vertices.end());
				}

				nextVertex++;
			} while (nextVertex != incidents);
		}

		queries.endQuery(origin);
	}
public:
	QueryBuilder() = default;

	QueryBuilder(const QueryBuilder& other) = delete;

	QueryBuilder(QueryBuilder&& other) noexcept :
		delaunay(std::move(other.delaunay)),
		buckets(std::move(other.buckets)) {}

	QueryBuilder& operator=(const QueryBuilder& other) = delete;

	QueryBuilder& operator=(QueryBuilder&& other) noexcept {
		this->delaunay = std::move(other.delaunay);
		this->buckets = std::move(other.buckets);
		return *this;
	}

	void insert(ServerID serverID, const GeographicPoint& point) {
		SphericalDelaunay::Vertex_handle vertex = this->delaunay.insert(point.toPoint());
		this->buckets[vertex].push_back(serverID);
	}

	template <typename PointGenerator>
	Queries build(PointGenerator generator, size_t points) {
		std::vector<Neighbour> vertices;
		VertexSet reached;

		Queries queries(points);

		vertices.reserve(this->delaunay.number_of_vertices());
		reached.reserve(this->delaunay.number_of_vertices());

		for (size_t count = 0; count < points; count++) {
			Point3 origin = generator();

			vertices.clear();
			reached.clear();

			SphericalDelaunay::Vertex_handle inserted = this->delaunay.insert(origin);

			if (!this->buckets.is_defined(inserted)) {
				auto incidents = this->delaunay.incident_vertices(inserted);
				auto nextVertex = incidents;

				if (incidents == nullptr) {
					this->delaunay.remove(inserted);
					throw std::invalid_argument("Delaunay triangulation is degenerate");
				}

				reached.insert(inserted);

				do {
					reached.insert(nextVertex);
					vertices.emplace_back(origin, this->delaunay, nextVertex);

					nextVertex++;
				} while (nextVertex != incidents);

				std::make_heap(vertices.begin(), vertices.end());

				this->dijkstraSearch(origin, vertices, reached, queries);

				this->delaunay.remove(inserted);
			} else {
				if (this->delaunay.incident_vertices(inserted) == nullptr)
					throw std::invalid_argument("Delaunay triangulation is degenerate");

				reached.insert(inserted);
				vertices.emplace_back(origin, this->delaunay, inserted);

				this->dijkstraSearch(origin, vertices, reached, queries);
			}
		}

		return queries;
	}
};

QueryBuilder parseServers(const std::string& inputFile) {
	simdjson::ondemand::parser jsonParser;
	simdjson::padded_string jsonString = simdjson::padded_string::load(inputFile);

	QueryBuilder builder;

	for (auto server : jsonParser.iterate(jsonString)) {
		ServerID serverID = server["server_id"].get_int64();
		std::string_view lat = server["latitude"];
		std::string_view lon = server["longtitude"];
		builder.insert(serverID, GeographicPoint(lat, lon));
	}

	return builder;
}

size_t pruneQueries(const Queries& queries, std::vector<GeographicPoint>& result) {
	absl::flat_hash_set<ServerID> covered;
	std::vector<std::vector<size_t>> buckets;

	for (size_t queryIndex = 0; queryIndex < queries.size(); queryIndex++) {
		size_t querySize = queries[queryIndex].size();

		if (querySize >= buckets.size())
			buckets.resize(querySize + 1);

		if (querySize > 0)
			buckets[querySize].push_back(queryIndex);
	}

	if (buckets.empty())
		return 0;

	for (size_t bucketIndex = buckets.size() - 1; bucketIndex > 0; bucketIndex--) {
		for (size_t queryIndex : buckets[bucketIndex]) {
			Queries::Query query = queries[queryIndex];

			size_t actualSize = query.size();

			for (ServerID serverID : query)
				if (covered.contains(serverID))
					actualSize--;

			if (actualSize == bucketIndex) {
				for (ServerID serverID : query)
					covered.insert(serverID);

				result.emplace_back(query.getLocation());
			} else if (actualSize > 0) {
				buckets[actualSize].push_back(queryIndex);
			}
		}
	}

	return covered.size();
}

namespace simdjson {
	template <typename builder_type>
	void tag_invoke(serialize_tag, builder_type& builder, const GeographicPoint& point) {
		builder.start_object();
		builder.append_key_value("latitude", point.getLatitude());
		builder.append_comma();
		builder.append_key_value("longtitude", point.getLongtitude());
		builder.end_object();
	}
}

int main(int argc, const char** argv) {
	if (argc != 3) {
		std::cout << "Usage: " << argv[0] << " [input file] [output file]" << std::endl;
		return 1;
	}

	QueryBuilder builder = parseServers(argv[1]);
	Queries queries = builder.build(RandomPointGenerator(), 1000000);

	std::vector<GeographicPoint> pruned;
	size_t covered = pruneQueries(queries, pruned);

	std::cout << "Covered " << covered << " servers using " << pruned.size() << " search queries." << std::endl;

	std::ofstream queryPlan(argv[2]);
	queryPlan << simdjson::to_json(pruned) << std::endl;

	return 0;
}
