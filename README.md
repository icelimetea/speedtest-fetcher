# Speedtest server fetcher

A tool to fetch a non-exhaustive list of [Speedtest.net](https://www.speedtest.net/) servers (but still huge nonetheless).

## Usage

`fetch_servers` Python script is used to fetch servers. Without a query plan (see `Query planner` section below) fetcher is forced to use API that does not report some additional information about servers, like their country.

You can use `fetch_servers` in the following way:

```sh
$ ./fetch_servers --output servers.json                           # Fetches a server list without some info
```

## Query planner

Fetcher can also retrieve servers' countries and HTTPS support, but it needs a query plan for that. Query planner builds a plan using previously fetched server list (that is, you should fetch the server list without a plan first).

You can use query planner together with fetcher like so:

```sh
$ ./fetch_servers --output servers.json                           # Fetch a preliminary server list (for query planner)
$ ./query_planner servers.json plan.json                          # Generate a query plan and save it to plan.json
$ ./fetch_servers --plan plan.json --output servers_extra.json    # Fetch servers using a plan
```

Query planner is written in C++ and depends on:
* abseil
* CGAL
* simdjson

It can be built using CMake:

```sh
$ git clone 'https://github.com/icelimetea/speedtest-fetcher.git'
$ cd speedtest-fetcher
$ cmake -DCMAKE_BUILD_TYPE=Release .
$ cmake --build .
```
