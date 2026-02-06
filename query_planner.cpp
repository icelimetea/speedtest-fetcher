#include <cstdint>
#include <cmath>
#include <numbers>
#include <vector>
#include <utility>
#include <algorithm>
#include <charconv>
#include <fstream>

#include <absl/flags/flag.h>
#include <absl/flags/parse.h>
#include <absl/flags/usage.h>
#include <absl/random/random.h>
#include <absl/container/flat_hash_set.h>
#include <absl/container/flat_hash_map.h>

#include <simdjson.h>

#include <CGAL/Unique_hash_map.h>
#include <CGAL/spatial_sort_on_sphere.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Delaunay_triangulation_on_sphere_traits_2.h>
#include <CGAL/Delaunay_triangulation_on_sphere_2.h>

using LinearKernel = CGAL::Exact_predicates_inexact_constructions_kernel;

using Point3 = LinearKernel::Point_3;

using ServerID = int32_t;

constexpr size_t RANDOM_POINTS_COUNT = 1 << 20;

class Angle {
private:
	double cos;
public:
	Angle(LinearKernel::FT radians) :
		cos(std::cos(radians)) {}

	Angle(const Point3& p1, const Point3& p2) :
		cos(p1.x() * p2.x() + p1.y() * p2.y() + p1.z() * p2.z()) {}

	LinearKernel::FT cosine() const {
		return this->cos;
	}

	friend bool operator<(const Angle& lhs, const Angle& rhs) {
		return lhs.cos > rhs.cos;
	}

	friend bool operator>(const Angle& lhs, const Angle& rhs) {
		return lhs.cos < rhs.cos;
	}
};

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

	LinearKernel::FT getLatitude() const {
		return this->lat;
	}

	LinearKernel::FT getLongtitude() const {
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
	absl::InsecureBitGen rng;
public:
	Point3 operator()() {
		double x = absl::Gaussian<double>(this->rng);
		double y = absl::Gaussian<double>(this->rng);
		double z = absl::Gaussian<double>(this->rng);

		double length = std::sqrt(x * x + y * y + z * z);

		return Point3(x, y, z, length);
	}
};

class Queries {
private:
	struct QueryTag {
		Point3 location;
		size_t size;
	};

	union Item {
		static constexpr size_t NUM_SERVERS = sizeof(QueryTag) / sizeof(ServerID);

		QueryTag tag;
		ServerID servers[NUM_SERVERS];

		Item() {}

		Item(const Point3& location) : tag(location, 0) {}
	};

	using ItemIterator = std::vector<Item>::const_iterator;

	std::vector<Item> items;

	size_t lastQueryIndex = 0;
public:
	static constexpr size_t SHORT_RANGE_SERVERS = 100;
	static constexpr size_t LONG_RANGE_SERVERS = 20;
	static constexpr size_t MAX_SERVERS_PER_RANGE = std::max(SHORT_RANGE_SERVERS, LONG_RANGE_SERVERS);

	class ServerIterator {
	private:
		ItemIterator itemIt;
		size_t index;
	public:
		using difference_type = std::ptrdiff_t;
		using value_type = ServerID;
		using reference = const value_type&;
		using pointer = const value_type*;
		using iterator_category = std::forward_iterator_tag;

		ServerIterator() = default;

		ServerIterator(const ServerIterator& other) = default;

		ServerIterator(ItemIterator itemIterator, size_t serverIndex) :
			itemIt(itemIterator + serverIndex / Item::NUM_SERVERS),
			index(serverIndex % Item::NUM_SERVERS) {}

		const ServerID& operator*() const {
			return this->itemIt->servers[this->index];
		}

		const ServerID* operator->() const {
			return &this->itemIt->servers[this->index];
		}

		ServerIterator& operator++() {
			this->itemIt += (this->index + 1) / Item::NUM_SERVERS;
			this->index = (this->index + 1) % Item::NUM_SERVERS;
			return *this;
		}

		ServerIterator operator++(int) {
			ServerIterator it = *this;
			++*this;
			return it;
		}

		bool operator==(const ServerIterator& other) const {
			return this->itemIt == other.itemIt && this->index == other.index;
		}
	};

	class Query {
	private:
		ItemIterator itemIt;
	public:
		Query(ItemIterator itemIterator) : itemIt(itemIterator) {}

		const Point3& getLocation() const {
			return this->itemIt->tag.location;
		}

		size_t size() const {
			return this->itemIt->tag.size;
		}

		ServerIterator begin() const {
			return ServerIterator(this->itemIt + 1, 0);
		}

		ServerIterator end() const {
			return ServerIterator(this->itemIt + 1, this->itemIt->tag.size);
		}
	};

	class QueryIterator {
	private:
		ItemIterator itemIt;
	public:
		using difference_type = std::ptrdiff_t;
		using value_type = Query;
		using reference = value_type;
		using pointer = void;
		using iterator_category = std::input_iterator_tag;

		QueryIterator(const QueryIterator& other) = default;

		QueryIterator(ItemIterator itemIterator) : itemIt(itemIterator) {}

		Query operator*() const {
			return Query(this->itemIt);
		}

		QueryIterator& operator++() {
			this->itemIt += (this->itemIt->tag.size + Item::NUM_SERVERS - 1) / Item::NUM_SERVERS + 1;
			return *this;
		}

		QueryIterator operator++(int) {
			QueryIterator it = *this;
			++*this;
			return it;
		}

		bool operator==(const QueryIterator& other) const {
			return this->itemIt == other.itemIt;
		}
	};

	Queries(size_t queryCount) {
		this->items.reserve((1 + (MAX_SERVERS_PER_RANGE + Item::NUM_SERVERS - 1) / Item::NUM_SERVERS) * queryCount);
	}

	Queries(const Queries& other) = delete;

	Queries& operator=(const Queries& other) = delete;

	void beginQuery(const Point3& location) {
		this->items.emplace_back(location);
		this->lastQueryIndex = this->items.size() - 1;
	}

	void insertServer(ServerID serverID) {
		size_t serverIndex = (this->items[this->lastQueryIndex].tag.size++) % Item::NUM_SERVERS;

		if (serverIndex == 0)
			this->items.emplace_back();

		this->items.back().servers[serverIndex] = serverID;
	}

	QueryIterator begin() const {
		return QueryIterator(this->items.begin());
	}

	QueryIterator end() const {
		return QueryIterator(this->items.end());
	}
};

class QueryBuilder {
private:
	using SphericalDelaunayTraits = CGAL::Delaunay_triangulation_on_sphere_traits_2<LinearKernel>;
	using SphericalDelaunay = CGAL::Delaunay_triangulation_on_sphere_2<SphericalDelaunayTraits>;

	using VertexHandle = SphericalDelaunay::Vertex_handle;
	using FaceHandle = SphericalDelaunay::Face_handle;

	using VertexSet = absl::flat_hash_set<VertexHandle>;
	using VertexMap = CGAL::Unique_hash_map<VertexHandle, std::vector<ServerID>>;

	SphericalDelaunay delaunay;
	VertexMap buckets;

	static constexpr double EARTH_RADIUS_MILES = 3963.1676;

	inline static const Angle SHORT_RANGE_ANGLE = Angle(30.0 / EARTH_RADIUS_MILES);
	inline static const Angle LONG_RANGE_ANGLE = Angle(2000.0 / EARTH_RADIUS_MILES);

	struct NoOpEdgeIterator {
		using difference_type = void;
		using value_type = void;
		using reference = void;
		using pointer = void;
		using iterator_category = std::output_iterator_tag;

		SphericalDelaunay::Edge operator*() {
			return SphericalDelaunay::Edge();
		}

		NoOpEdgeIterator& operator++() {
			return *this;
		}

		NoOpEdgeIterator operator++(int) {
			return *this;
		}
	};

	struct Neighbour {
		VertexHandle vertex;
		Angle distance;

		Neighbour(const Point3& origin, const SphericalDelaunay& delaunay, VertexHandle vertexHandle) :
			vertex(vertexHandle),
			distance(origin, delaunay.point(vertexHandle)) {}

		friend bool operator<(const Neighbour& lhs, const Neighbour& rhs) {
			return lhs.distance < rhs.distance;
		}

		friend bool operator>(const Neighbour& lhs, const Neighbour& rhs) {
			return lhs.distance > rhs.distance;
		}
	};

	void dijkstraSearch(const Point3& origin,
			    std::vector<Neighbour>& vertices,
			    VertexSet& reached,
			    Queries& queries) const {
		queries.beginQuery(origin);

		size_t insertedServers = 0;

		while (!vertices.empty()) {
			const Neighbour& neighbour = vertices.front();

			size_t limit;

			if (neighbour.distance < SHORT_RANGE_ANGLE) {
				limit = Queries::SHORT_RANGE_SERVERS;
			} else if (neighbour.distance < LONG_RANGE_ANGLE) {
				limit = Queries::LONG_RANGE_SERVERS;
			} else {
				return;
			}

			for (ServerID serverID : this->buckets[neighbour.vertex]) {
				if (insertedServers >= limit)
					return;

				queries.insertServer(serverID);

				insertedServers++;
			}

			auto incidents = this->delaunay.incident_vertices(neighbour.vertex);
			auto nextVertex = incidents;

			std::pop_heap(vertices.begin(), vertices.end(), std::greater{});
			vertices.pop_back();

			do {
				if (reached.insert(nextVertex).second) {
					vertices.emplace_back(origin, this->delaunay, nextVertex);
					std::push_heap(vertices.begin(), vertices.end(), std::greater{});
				}

				nextVertex++;
			} while (nextVertex != incidents);
		}
	}

	void build(Queries& queries, const std::vector<Point3>& searchPoints) const {
		std::vector<Neighbour> vertices;
		vertices.reserve(this->delaunay.number_of_vertices());

		std::vector<FaceHandle> faces;
		faces.reserve(this->delaunay.number_of_faces());

		VertexSet reached;
		reached.reserve(this->delaunay.number_of_vertices());

		FaceHandle loc;

		for (const Point3& origin : searchPoints) {
			SphericalDelaunay::Locate_type lt;
			int li;
			loc = this->delaunay.locate(origin, lt, li, loc);

			if (lt != SphericalDelaunay::Locate_type::VERTEX && lt != SphericalDelaunay::Locate_type::TOO_CLOSE) {
				this->delaunay.get_conflicts_and_boundary(origin, std::back_inserter(faces), NoOpEdgeIterator(), loc);

				for (FaceHandle face : faces) {
					face->tds_data().clear();

					for (int i = 0; i < 3; i++) {
						VertexHandle vertex = face->vertex(i);

						if (reached.insert(vertex).second)
							vertices.emplace_back(origin, this->delaunay, vertex);
					}
				}

				std::make_heap(vertices.begin(), vertices.end(), std::greater{});
			} else {
				VertexHandle found = loc->vertex(li);

				reached.insert(found);
				vertices.emplace_back(origin, this->delaunay, found);
			}

			this->dijkstraSearch(origin, vertices, reached, queries);

			vertices.clear();
			faces.clear();
			reached.clear();
		}
	}
public:
	template <typename ServerIt>
	QueryBuilder(ServerIt begin, ServerIt end) {
		for (ServerIt it = begin; it != end; ++it) {
			const std::pair<ServerID, Point3>& server = *it;
			VertexHandle vertex = this->delaunay.insert(server.second);
			this->buckets[vertex].push_back(server.first);
		}

		if (this->delaunay.dimension() != 2)
			throw std::invalid_argument("Delaunay triangulation is degenerate");
	}

	QueryBuilder(const QueryBuilder& other) = delete;

	QueryBuilder& operator=(const QueryBuilder& other) = delete;

	template <typename PointGenerator>
	void build(Queries& queries, PointGenerator generator, size_t pointsCount) const {
		std::vector<Point3> points;
		points.reserve(pointsCount);

		for (size_t count = 0; count < pointsCount; count++)
			points.push_back(generator());

		CGAL::spatial_sort_on_sphere(points.begin(), points.end());

		this->build(queries, points);
	}
};

class ServerList {
private:
	using JsonIterator = simdjson::simdjson_result<simdjson::ondemand::array_iterator>;

	simdjson::padded_string jsonData;

	simdjson::ondemand::parser jsonParser;
	simdjson::ondemand::document jsonDocument;
public:
	class ServerListIterator {
	private:
		JsonIterator jsonIterator;
	public:
		using difference_type = std::ptrdiff_t;
		using value_type = std::pair<ServerID, Point3>;
		using reference = value_type;
		using pointer = void;
		using iterator_category = std::input_iterator_tag;

		ServerListIterator(const ServerListIterator& other) = default;

		ServerListIterator(const JsonIterator& jsonIterator) : jsonIterator(jsonIterator) {}

		std::pair<ServerID, Point3> operator*() {
			auto serverObj = *this->jsonIterator;

			ServerID serverID = serverObj["server_id"].get_int64();

			std::string_view lat = serverObj["latitude"];
			std::string_view lon = serverObj["longtitude"];

			return std::make_pair(serverID, GeographicPoint(lat, lon).toPoint());
		}

		ServerListIterator& operator++() {
			++this->jsonIterator;
			return *this;
		}

		bool operator==(const ServerListIterator& other) const {
			return this->jsonIterator == other.jsonIterator;
		}
	};

	ServerList(const std::string& inputFile) {
		this->jsonData = simdjson::padded_string::load(inputFile);
		this->jsonDocument = this->jsonParser.iterate(this->jsonData);
	}

	ServerListIterator begin() {
		return ServerListIterator(this->jsonDocument.begin());
	}

	ServerListIterator end() {
		return ServerListIterator(this->jsonDocument.end());
	}
};

size_t pruneQueries(std::vector<GeographicPoint>& result, const Queries& queries) {
	absl::flat_hash_set<ServerID> covered;
	std::vector<std::vector<Queries::Query>> buckets;

	for (Queries::Query query : queries) {
		size_t querySize = query.size();

		if (querySize >= buckets.size())
			buckets.resize(querySize + 1);

		if (querySize > 0)
			buckets[querySize].push_back(query);
	}

	if (buckets.empty())
		return 0;

	for (size_t bucketIndex = buckets.size() - 1; bucketIndex > 0; bucketIndex--) {
		for (Queries::Query query : buckets[bucketIndex]) {
			size_t actualSize = query.size();

			for (ServerID serverID : query)
				if (covered.contains(serverID))
					actualSize--;

			if (actualSize == bucketIndex) {
				for (ServerID serverID : query)
					covered.insert(serverID);

				result.emplace_back(query.getLocation());
			} else if (actualSize > 0) {
				buckets[actualSize].push_back(query);
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

void dumpQueries(const std::string& outputFile, const Queries& queries) {
	std::ofstream outputStream(outputFile);

	absl::flat_hash_map<ServerID, std::vector<size_t>> constraints;

	size_t queryIndex = 0;

	for (Queries::Query query : queries) {
		for (ServerID serverID : query)
			constraints[serverID].push_back(queryIndex);

		queryIndex++;
	}

	outputStream << "Minimize obj:";

	for (size_t varIndex = 0; varIndex < queryIndex; varIndex++)
		outputStream << " + x" << varIndex;

	outputStream << std::endl;

	outputStream << "Subject To" << std::endl;

	for (const auto& [serverID, queries] : constraints) {
		for (size_t varIndex : queries)
			outputStream << " + x" << varIndex;

		outputStream << " >= 1" << std::endl;
	}

	outputStream << "Binary";

	for (size_t varIndex = 0; varIndex < queryIndex; varIndex++)
		outputStream << " x" << varIndex;

	outputStream << std::endl;

	outputStream << "End" << std::endl;
}

void dumpPoints(const std::string& outputFile, const std::vector<GeographicPoint>& points) {
	std::ofstream outputStream(outputFile);
	outputStream << simdjson::to_json(points) << std::endl;
}

ABSL_FLAG(std::optional<std::string>, servers, std::nullopt, "Input file containing server list in JSON format");
ABSL_FLAG(std::optional<std::string>, plan, std::nullopt, "Output file for storing planned queries in JSON format");
ABSL_FLAG(std::optional<std::string>, queries, std::nullopt, "If specified, output file for storing unpruned queries as BIP");

int main(int argc, char** argv) {
	absl::SetProgramUsageMessage("Query planner for server fetcher");

	absl::ParseCommandLine(argc, argv);

	if (!absl::GetFlag(FLAGS_servers)) {
		std::cout << "Input server list is not provided. See --help for details." << std::endl;
		return 1;
	}

	if (!absl::GetFlag(FLAGS_plan)) {
		std::cout << "Output file name for planned queries is not provided. See --help for details." << std::endl;
		return 1;
	}

	std::string serversFile = absl::GetFlag(FLAGS_servers).value();
	std::string planFile = absl::GetFlag(FLAGS_plan).value();
	std::optional<std::string> queriesFile = absl::GetFlag(FLAGS_queries);

	ServerList serverList(serversFile);

	QueryBuilder builder(serverList.begin(), serverList.end());
	Queries queries(RANDOM_POINTS_COUNT);
	builder.build(queries, RandomPointGenerator(), RANDOM_POINTS_COUNT);

	if (queriesFile)
		dumpQueries(queriesFile.value(), queries);

	std::vector<GeographicPoint> pruned;
	size_t covered = pruneQueries(pruned, queries);

	std::cout << "Covered " << covered << " servers using " << pruned.size() << " search queries." << std::endl;

	dumpPoints(planFile, pruned);

	return 0;
}
