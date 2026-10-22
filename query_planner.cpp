#include <cassert>
#include <cstdint>
#include <cmath>
#include <bit>
#include <numbers>
#include <limits>
#include <array>
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

#include <simdjson.h>

#include <CGAL/spatial_sort_on_sphere.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Triangulation_vertex_base_with_info_2.h>
#include <CGAL/Delaunay_triangulation_on_sphere_traits_2.h>
#include <CGAL/Delaunay_triangulation_on_sphere_2.h>

using LinearKernel = CGAL::Exact_predicates_inexact_constructions_kernel;

using SphericalDelaunayTraits = CGAL::Delaunay_triangulation_on_sphere_traits_2<LinearKernel>;

using Point3 = LinearKernel::Point_3;

using ServerID = int32_t;

class Angle {
private:
	LinearKernel::FT cos;
public:
	Angle(LinearKernel::FT radians) {
		assert(0 <= radians && radians <= std::numbers::pi);

		this->cos = std::cos(radians);
	}

	Angle(const Point3& p1, const Point3& p2) {
		assert(SphericalDelaunayTraits().is_on_sphere(p1));
		assert(SphericalDelaunayTraits().is_on_sphere(p2));

		this->cos = p1.x() * p2.x() + p1.y() * p2.y() + p1.z() * p2.z();
	}

	LinearKernel::FT cosine() const {
		return this->cos;
	}

	bool operator<(const Angle& other) const {
		return this->cos > other.cos;
	}

	bool operator>(const Angle& other) const {
		return this->cos < other.cos;
	}
};

class GeographicPoint {
private:
	LinearKernel::FT lat;
	LinearKernel::FT lon;
public:
	GeographicPoint(std::string_view latitude, std::string_view longitude) {
		const char* lat_start = latitude.data();
		const char* lat_end = latitude.data() + latitude.size();

		const char* lon_start = longitude.data();
		const char* lon_end = longitude.data() + longitude.size();

		std::from_chars_result lat_result = std::from_chars(lat_start, lat_end, this->lat);
		std::from_chars_result lon_result = std::from_chars(lon_start, lon_end, this->lon);

		if (lat_result.ec != std::errc() || lat_result.ptr != lat_end)
			throw std::invalid_argument("Invalid latitude");

		if (lon_result.ec != std::errc() || lon_result.ptr != lon_end)
			throw std::invalid_argument("Invalid longitude");

		if (!std::isfinite(this->lat))
			throw std::invalid_argument("Latitude should be a finite floating-point value");

		if (!std::isfinite(this->lon))
			throw std::invalid_argument("Longitude should be a finite floating-point value");
	}

	GeographicPoint(const Point3& point) {
		assert(SphericalDelaunayTraits().is_on_sphere(point));

		this->lat = std::asin(point.z()) * 180 / std::numbers::pi;
		this->lon = std::atan2(point.y(), point.x()) * 180 / std::numbers::pi;
	}

	LinearKernel::FT getLatitude() const {
		return this->lat;
	}

	LinearKernel::FT getLongitude() const {
		return this->lon;
	}

	Point3 toPoint() const {
		LinearKernel::FT latRad = this->lat * std::numbers::pi / 180;
		LinearKernel::FT lonRad = this->lon * std::numbers::pi / 180;

		LinearKernel::FT z = std::sin(latRad);
		LinearKernel::FT h = std::cos(latRad);
		LinearKernel::FT x = h * std::cos(lonRad);
		LinearKernel::FT y = h * std::sin(lonRad);

		return Point3(x, y, z);
	}
};

class RandomSpherePointGenerator {
private:
	static constexpr LinearKernel::RT MIN_LENGTH = 1.0 / 1024.0;

	absl::InsecureBitGen rng;
public:
	Point3 operator()() {
		LinearKernel::RT x;
		LinearKernel::RT y;
		LinearKernel::RT z;
		LinearKernel::RT w;

		do {
			x = absl::Gaussian<LinearKernel::RT>(this->rng);
			y = absl::Gaussian<LinearKernel::RT>(this->rng);
			z = absl::Gaussian<LinearKernel::RT>(this->rng);
			w = std::hypot(x, y, z);
		} while (w < MIN_LENGTH);

		return Point3(x, y, z, w);
	}
};

template <typename Metadata, typename Element>
class PackedVector {
private:
	struct ItemTag {
		Metadata entryMeta;
		size_t entrySize;
	};

	union Item {
		static constexpr size_t NUM_ELEMENTS = std::bit_ceil((sizeof(ItemTag) + sizeof(Element) - 1) / sizeof(Element));

		ItemTag tag;
		Element data[NUM_ELEMENTS];

		Item() {}

		Item(const Metadata& entryMeta, size_t entrySize) :
			tag(entryMeta, entrySize) {}
	};

	using ItemIterator = std::vector<Item>::const_iterator;

	std::vector<Item> items;

	size_t lastItemIndex;
public:
	class ElementIterator {
	private:
		ItemIterator iter;
		size_t index;
	public:
		using difference_type = std::ptrdiff_t;
		using value_type = Element;
		using reference = const value_type&;
		using pointer = const value_type*;
		using iterator_category = std::forward_iterator_tag;

		ElementIterator() = default;

		ElementIterator(const ElementIterator& other) = default;

		ElementIterator(ItemIterator itemIterator, size_t elementIndex) :
			iter(itemIterator + elementIndex / Item::NUM_ELEMENTS),
			index(elementIndex % Item::NUM_ELEMENTS) {}

		const Element& operator*() const {
			return this->iter->data[this->index];
		}

		const Element* operator->() const {
			return &this->iter->data[this->index];
		}

		ElementIterator& operator++() {
			this->iter += (this->index + 1) / Item::NUM_ELEMENTS;
			this->index = (this->index + 1) % Item::NUM_ELEMENTS;
			return *this;
		}

		ElementIterator operator++(int) {
			ElementIterator it = *this;
			++*this;
			return it;
		}

		bool operator==(const ElementIterator& other) const {
			return this->iter == other.iter && this->index == other.index;
		}
	};

	class Entry {
	private:
		ItemIterator iter;
	public:
		Entry(ItemIterator itemIterator) : iter(itemIterator) {}

		const Metadata& metadata() const {
			return this->iter->tag.entryMeta;
		}

		size_t size() const {
			return this->iter->tag.entrySize;
		}

		ElementIterator begin() const {
			return ElementIterator(this->iter + 1, 0);
		}

		ElementIterator end() const {
			return ElementIterator(this->iter + 1, this->iter->tag.entrySize);
		}
	};

	class EntryIterator {
	private:
		ItemIterator iter;
	public:
		using difference_type = std::ptrdiff_t;
		using value_type = Entry;
		using reference = value_type;
		using pointer = void;
		using iterator_category = std::input_iterator_tag;

		EntryIterator(const EntryIterator& other) = default;

		EntryIterator(ItemIterator itemIterator) : iter(itemIterator) {}

		Entry operator*() const {
			return Entry(this->iter);
		}

		EntryIterator& operator++() {
			this->iter += 1 + (this->iter->tag.entrySize + Item::NUM_ELEMENTS - 1) / Item::NUM_ELEMENTS;
			return *this;
		}

		EntryIterator operator++(int) {
			EntryIterator it = *this;
			++*this;
			return it;
		}

		bool operator==(const EntryIterator& other) const {
			return this->iter == other.iter;
		}
	};

	PackedVector(size_t entryCount, size_t entrySize) {
		this->items.reserve((1 + (entrySize + Item::NUM_ELEMENTS - 1) / Item::NUM_ELEMENTS) * entryCount);
	}

	PackedVector(const PackedVector& other) = delete;

	PackedVector& operator=(const PackedVector& other) = delete;

	void beginEntry(const Metadata& metadata) {
		this->items.emplace_back(metadata, 0);
		this->lastItemIndex = this->items.size() - 1;
	}

	void insertElement(const Element& element) {
		assert(!this->items.empty());

		size_t elementIndex = (this->items[this->lastItemIndex].tag.entrySize++) % Item::NUM_ELEMENTS;

		if (elementIndex == 0)
			this->items.emplace_back();

		this->items.back().data[elementIndex] = element;
	}

	EntryIterator begin() const {
		return EntryIterator(this->items.begin());
	}

	EntryIterator end() const {
		return EntryIterator(this->items.end());
	}
};

class QueryBuilder {
public:
	static constexpr size_t MAX_SHORT_RANGE_SERVERS = 100;
	static constexpr size_t MAX_LONG_RANGE_SERVERS = 20;
	static constexpr size_t MAX_SERVERS_PER_RANGE = std::max(MAX_SHORT_RANGE_SERVERS, MAX_LONG_RANGE_SERVERS);
private:
	static constexpr LinearKernel::FT EARTH_RADIUS_MILES = 3963.1676;

	static constexpr LinearKernel::FT SHORT_RANGE_MILES = 30.0;
	static constexpr LinearKernel::FT LONG_RANGE_MILES = 2000.0;

	inline static const Angle SHORT_RANGE_ANGLE = Angle(SHORT_RANGE_MILES / EARTH_RADIUS_MILES);
	inline static const Angle LONG_RANGE_ANGLE = Angle(LONG_RANGE_MILES / EARTH_RADIUS_MILES);

	inline static const std::array<Point3, 6> EPHEMERAL_POINTS = {
		Point3(-1,  0,  0),
		Point3( 0, -1,  0),
		Point3( 0,  0, -1),
		Point3( 1,  0,  0),
		Point3( 0,  1,  0),
		Point3( 0,  0,  1)
	};

	struct VertexInfo {
		bool reached = false;
		std::vector<ServerID> servers;
	};

	using VertexBase = CGAL::Triangulation_on_sphere_vertex_base_2<SphericalDelaunayTraits>;
	using VertexBaseWithID = CGAL::Triangulation_vertex_base_with_info_2<VertexInfo, SphericalDelaunayTraits, VertexBase>;

	using FaceBase = CGAL::Triangulation_on_sphere_face_base_2<SphericalDelaunayTraits>;

	using SphericalTDS = CGAL::Triangulation_data_structure_2<VertexBaseWithID, FaceBase>;
	using SphericalDelaunay = CGAL::Delaunay_triangulation_on_sphere_2<SphericalDelaunayTraits, SphericalTDS>;

	using VertexHandle = SphericalDelaunay::Vertex_handle;
	using FaceHandle = SphericalDelaunay::Face_handle;

	using LocateType = SphericalDelaunay::Locate_type;

	SphericalDelaunay delaunay;

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

		Neighbour(const Point3& origin, VertexHandle vertex) :
			vertex(vertex),
			distance(origin, vertex->point()) {}

		bool operator<(const Neighbour& other) const {
			return this->distance < other.distance;
		}

		bool operator>(const Neighbour& other) const {
			return this->distance > other.distance;
		}
	};

	void dijkstraSearch(
			const Point3& origin,
			std::vector<Neighbour>& vertices,
			std::vector<VertexHandle>& clearList,
			PackedVector<Point3, ServerID>& queries
	) const {
		queries.beginEntry(origin);

		size_t insertedServers = 0;

		while (!vertices.empty()) {
			const Neighbour& neighbour = vertices.front();

			size_t limit;

			if (neighbour.distance < SHORT_RANGE_ANGLE) {
				limit = MAX_SHORT_RANGE_SERVERS;
			} else if (neighbour.distance < LONG_RANGE_ANGLE) {
				limit = MAX_LONG_RANGE_SERVERS;
			} else {
				return;
			}

			for (ServerID serverID : neighbour.vertex->info().servers) {
				if (insertedServers >= limit)
					return;

				queries.insertElement(serverID);

				insertedServers++;
			}

			auto incidents = this->delaunay.incident_vertices(neighbour.vertex);
			auto nextVertex = incidents;

			std::pop_heap(vertices.begin(), vertices.end(), std::greater{});
			vertices.pop_back();

			do {
				if (!nextVertex->info().reached) {
					nextVertex->info().reached = true;

					clearList.push_back(nextVertex);

					vertices.emplace_back(origin, nextVertex);
					std::push_heap(vertices.begin(), vertices.end(), std::greater{});
				}

				nextVertex++;
			} while (nextVertex != incidents);
		}
	}

	void build(PackedVector<Point3, ServerID>& queries, const std::vector<Point3>& searchPoints) const {
		std::vector<Neighbour> vertices;
		vertices.reserve(this->delaunay.number_of_vertices());

		std::vector<FaceHandle> faces;
		faces.reserve(this->delaunay.number_of_faces());

		std::vector<VertexHandle> clearList;
		clearList.reserve(this->delaunay.number_of_vertices());

		FaceHandle loc;

		for (const Point3& origin : searchPoints) {
			LocateType lt;
			int li;
			loc = this->delaunay.locate(origin, lt, li, loc);

			assert(loc != FaceHandle());

			if (lt == LocateType::VERTEX || lt == LocateType::TOO_CLOSE) {
				VertexHandle found = loc->vertex(li);

				found->info().reached = true;

				clearList.push_back(found);
				vertices.emplace_back(origin, found);
			} else {
				this->delaunay.get_conflicts_and_boundary(origin, std::back_inserter(faces), NoOpEdgeIterator(), loc);

				for (FaceHandle face : faces) {
					face->tds_data().clear();

					for (int i = 0; i < 3; i++) {
						VertexHandle vertex = face->vertex(i);

						if (!vertex->info().reached) {
							vertex->info().reached = true;

							clearList.push_back(vertex);
							vertices.emplace_back(origin, vertex);
						}
					}
				}

				std::make_heap(vertices.begin(), vertices.end(), std::greater{});
			}

			this->dijkstraSearch(origin, vertices, clearList, queries);

			while (!clearList.empty()) {
				clearList.back()->info().reached = false;
				clearList.pop_back();
			}

			vertices.clear();
			faces.clear();
		}
	}
public:
	template <typename ServerIt>
	QueryBuilder(ServerIt begin, ServerIt end) {
		for (const Point3& point : EPHEMERAL_POINTS)
			this->delaunay.insert(point);

		for (ServerIt it = begin; it != end; ++it) {
			const std::pair<ServerID, GeographicPoint>& server = *it;

			VertexHandle vertex = this->delaunay.insert(server.second.toPoint());

			assert(vertex != VertexHandle());

			vertex->info().servers.push_back(server.first);
		}

		assert(this->delaunay.dimension() == 2);
	}

	QueryBuilder(const QueryBuilder& other) = delete;

	QueryBuilder& operator=(const QueryBuilder& other) = delete;

	template <typename SpherePointGenerator>
	void build(PackedVector<Point3, ServerID>& queries, SpherePointGenerator generator, size_t pointsCount) const {
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

	simdjson::ondemand::parser jsonParser;

	simdjson::padded_string jsonData;
	simdjson::ondemand::document jsonDocument;
public:
	class ServerListIterator {
	private:
		JsonIterator iter;
	public:
		using difference_type = std::ptrdiff_t;
		using value_type = std::pair<ServerID, GeographicPoint>;
		using reference = value_type;
		using pointer = void;
		using iterator_category = std::input_iterator_tag;

		ServerListIterator(const ServerListIterator& other) = default;

		ServerListIterator(const JsonIterator& jsonIterator) : iter(jsonIterator) {}

		std::pair<ServerID, GeographicPoint> operator*() {
			auto serverObj = *this->iter;

			int64_t uncheckedServerID = serverObj["server_id"].get_int64();

			if (uncheckedServerID < 0 || uncheckedServerID > std::numeric_limits<ServerID>::max())
				throw std::invalid_argument("Server ID is out of range");

			ServerID serverID = static_cast<ServerID>(uncheckedServerID);

			std::string_view lat = serverObj["latitude"];
			std::string_view lon = serverObj["longitude"];

			return std::make_pair(serverID, GeographicPoint(lat, lon));
		}

		ServerListIterator& operator++() {
			++this->iter;
			return *this;
		}

		bool operator==(const ServerListIterator& other) const {
			return this->iter == other.iter;
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

namespace simdjson {
	template <typename builder_type>
	void tag_invoke(serialize_tag, builder_type& builder, const GeographicPoint& point) {
		builder.start_object();
		builder.append_key_value("latitude", point.getLatitude());
		builder.append_comma();
		builder.append_key_value("longitude", point.getLongitude());
		builder.end_object();
	}
}

static size_t pruneQueries(std::vector<GeographicPoint>& result, const PackedVector<Point3, ServerID>& queries) {
	absl::flat_hash_set<ServerID> covered;

	std::vector<std::vector<PackedVector<Point3, ServerID>::Entry>> buckets;

	for (PackedVector<Point3, ServerID>::Entry query : queries) {
		size_t querySize = query.size();

		if (querySize > buckets.size())
			buckets.resize(querySize);

		if (querySize > 0)
			buckets[querySize - 1].push_back(query);
	}

	if (buckets.empty())
		return 0;

	for (size_t bucketSize = buckets.size(); bucketSize > 0; bucketSize--) {
		for (PackedVector<Point3, ServerID>::Entry query : buckets[bucketSize - 1]) {
			size_t actualSize = 0;

			for (ServerID serverID : query)
				if (!covered.contains(serverID))
					actualSize++;

			if (actualSize == bucketSize) {
				covered.insert(query.begin(), query.end());

				result.emplace_back(query.metadata());
			} else if (actualSize > 0) {
				buckets[actualSize - 1].push_back(query);
			}
		}
	}

	return covered.size();
}

static void dumpSetCover(const std::string& outputFile, const PackedVector<Point3, ServerID>& queries) {
	std::ofstream outputStream(outputFile);

	absl::flat_hash_set<ServerID> servers;

	for (PackedVector<Point3, ServerID>::Entry query : queries)
		servers.insert(query.begin(), query.end());

	outputStream << "NAME SET_COVER" << "\n";

	outputStream << "OBJSENSE MIN" << "\n";

	outputStream << "ROWS" << "\n";

	outputStream
		<< " "
		<< "N" << " "
		<< "QUERIES_COUNT" << "\n";

	for (ServerID serverID : servers) {
		outputStream
			<< " "
			<< "G" << " "
			<< "SERVER_" << serverID << "\n";
	}

	outputStream << "COLUMNS" << "\n";

	size_t queryIndex = 0;
	for (PackedVector<Point3, ServerID>::Entry query : queries) {
		outputStream
			<< "*" << " "
			<< "QUERY_" << queryIndex << " "
			<< query.metadata() << "\n";

		outputStream
			<< " "
			<< " "
			<< "QUERY_" << queryIndex << " "
			<< "QUERIES_COUNT" << " "
			<< "1" << "\n";

		for (ServerID serverID : query) {
			outputStream
				<< " "
				<< " "
				<< "QUERY_" << queryIndex << " "
				<< "SERVER_" << serverID << " "
				<< "1" << "\n";
		}

		queryIndex++;
	}

	outputStream << "RHS" << "\n";

	for (ServerID serverID : servers) {
		outputStream
			<< " "
			<< " "
			<< "RHS_VECTOR" << " "
			<< "SERVER_" << serverID << " "
			<< "1" << "\n";
	}

	outputStream << "BOUNDS" << "\n";

	for (size_t columnIndex = 0; columnIndex < queryIndex; columnIndex++) {
		outputStream
			<< " "
			<< "BV" << " "
			<< "BOUNDS_VECTOR" << " "
			<< "QUERY_" << columnIndex << "\n";
	}

	outputStream << "ENDATA" << std::endl;
}

static void dumpPoints(const std::string& outputFile, const std::vector<GeographicPoint>& points) {
	std::ofstream outputStream(outputFile);
	outputStream << simdjson::to_json(points) << std::endl;
}

ABSL_FLAG(std::optional<std::string>, servers, std::nullopt, "Input file containing server list in JSON format");
ABSL_FLAG(std::optional<std::string>, plan, std::nullopt, "Output file for storing planned queries in JSON format");
ABSL_FLAG(std::optional<std::string>, setcover, std::nullopt, "If specified, output file for storing the set cover problem instance as a BIP (using CPLEX MPS format)");
ABSL_FLAG(size_t, points, 1 << 20, "Number of points to sample");

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
	std::optional<std::string> setCoverFile = absl::GetFlag(FLAGS_setcover);
	size_t pointsCount = absl::GetFlag(FLAGS_points);

	ServerList serverList(serversFile);

	QueryBuilder builder(serverList.begin(), serverList.end());
	PackedVector<Point3, ServerID> queries(pointsCount, QueryBuilder::MAX_SERVERS_PER_RANGE);

	builder.build(queries, RandomSpherePointGenerator(), pointsCount);

	if (setCoverFile.has_value())
		dumpSetCover(setCoverFile.value(), queries);

	std::vector<GeographicPoint> pruned;
	size_t covered = pruneQueries(pruned, queries);

	std::cout << "Covered " << covered << " servers using " << pruned.size() << " search queries." << std::endl;

	dumpPoints(planFile, pruned);

	return 0;
}
