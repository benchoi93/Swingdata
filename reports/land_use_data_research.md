# Land Use Data Sources for E-Scooter Speed Governance Study

**Date:** 2026-03-13
**Purpose:** Evaluate data sources for adding land use variables at trip-origin level across 52 South Korean cities.
**Context:** 19.5M scooter trips (Feb-Nov 2023) with start_lat/start_lon coordinates. Need to classify each trip origin as residential, commercial, industrial, mixed-use, or similar.

---

## 1. Korean Government Data Sources

### 1A. MOLIT Land Use Status Map (국토교통부 토지이용현황도)

**Source:** [data.go.kr Dataset 3049890](https://www.data.go.kr/data/3049890/fileData.do) and [Dataset 15059724 (2024 update)](https://www.data.go.kr/data/15059724/fileData.do)

- **Provider:** Ministry of Land, Infrastructure and Transport (국토교통부) via National Geographic Information Institute (국토지리정보원)
- **Format:** SHP (Shapefile)
- **Classification:** 6-level system: Urban (도시), Agriculture/paddies (논), Agriculture/fields (밭), Forest-established (성림), Forest-unestablished (미성림), Industrial (공업), Natural/Cultural Heritage, Reserved
- **Resolution:** Parcel-level (필지별) -- very fine-grained
- **Coverage:** Nationwide (전국) -- covers all 52 study cities
- **Availability:** Open data, free download via data.go.kr
- **Last updated:** September 2024 (dataset 15059724)
- **Licensing:** Government open data license (공공누리)

**Assessment:**
- (+) Nationwide parcel-level coverage -- excellent spatial resolution
- (+) Official government source -- high credibility for publication
- (+) Free and open -- no barriers
- (-) The 6-level classification is land-COVER oriented (urban/forest/agriculture), not land-USE zoning (residential/commercial/industrial)
- (-) May require large file downloads per municipality; SHP format but manageable with geopandas
- (-) "Urban" is a single category -- does NOT distinguish residential from commercial from industrial
- **Verdict:** Not ideal for this study because it cannot distinguish between residential, commercial, and industrial within urban areas. The classification is too coarse for our needs.

### 1B. MOLIT Land Use Plan Information (국토교통부 토지이용계획정보)

**Source:** [data.go.kr Dataset 15045900](https://www.data.go.kr/data/15045900/fileData.do)

- **Provider:** MOLIT
- **Format:** CSV/SHP files organized by administrative district code and parcel number
- **Classification:** Zoning-based (용도지역) -- the Korean planning system classifies land into:
  - **Residential (주거지역):** Exclusive Residential (전용주거), General Residential 1/2/3 (일반주거 1-3종), Semi-Residential (준주거)
  - **Commercial (상업지역):** Central Commercial (중심상업), General Commercial (일반상업), Neighborhood Commercial (근린상업), Distribution Commercial (유통상업)
  - **Industrial (공업지역):** Exclusive Industrial (전용공업), General Industrial (일반공업), Semi-Industrial (준공업)
  - **Green (녹지지역):** Conservation Green (보전녹지), Production Green (생산녹지), Natural Green (자연녹지)
  - **Management (관리지역), Agricultural/Forest (농림지역), Natural Environment Conservation (자연환경보전지역)**
- **Resolution:** Parcel-level
- **Coverage:** Nationwide
- **Availability:** Open data via data.go.kr
- **Last updated:** July 2024

**Assessment:**
- (+) **Exactly the classification we need** -- residential vs commercial vs industrial vs green
- (+) Fine-grained subcategories (e.g., General Residential 1/2/3 density levels)
- (+) Nationwide coverage, parcel-level resolution
- (+) Free and open
- (-) Data organized by parcel ID (PNU code), not as spatial polygons -- may require joining with cadastral boundary data to create spatial polygons
- (-) Large dataset (millions of parcels nationwide)
- (-) Requires understanding of Korean parcel numbering (PNU) system
- **Verdict:** BEST Korean government source for this study. Provides the exact residential/commercial/industrial classification. Main challenge is spatial joining -- need cadastral boundary polygons to match GPS coordinates.

### 1C. 토지이음 (Land E-um, eum.go.kr) -- Land Use Regulation Portal

**Source:** [eum.go.kr](https://www.eum.go.kr/)

- **Provider:** MOLIT
- **Format:** Web viewer + OpenAPI for individual parcel lookup; CSV downloads for notification data
- **Classification:** Full zoning (용도지역/지구/구역) information per parcel
- **Coverage:** Nationwide
- **Availability:** Free web access; API requires registration

**Assessment:**
- (+) Comprehensive zoning information per parcel
- (-) Designed for individual parcel lookup, NOT bulk download
- (-) No obvious bulk SHP/spatial download for nationwide coverage
- (-) API appears to be per-parcel query -- impractical for 19.5M trip origins
- **Verdict:** Useful reference but impractical for bulk spatial analysis.

### 1D. SGIS (통계지리정보서비스, KOSTAT)

**Source:** [sgis.kostat.go.kr](https://sgis.kostat.go.kr/) and [data.go.kr Dataset 15021230](https://www.data.go.kr/data/15021230/openapi.do)

- **Provider:** Statistics Korea (통계청)
- **Format:** OpenAPI + downloadable spatial files (SHP via QGIS/Python/R)
- **Data available:**
  - Population/household/housing census data at census tract (집계구) and grid (격자) levels
  - Business census data (사업체조사) -- establishment counts by industry
  - Census tract boundaries and grid boundaries (100m grids)
  - Time series: 2000-2023
- **Resolution:** Census tract (~500m) or 100m grid
- **Coverage:** Nationwide
- **Availability:** Free with API key registration; some data requires application via portal

**Assessment:**
- (+) Population and business establishment density -- excellent proxy for land use characterization
- (+) 100m grid resolution is sufficient for trip-origin classification
- (+) Nationwide coverage, well-maintained by national statistics office
- (+) Business census can distinguish commercial/industrial/service establishments
- (+) Python-compatible via API or downloaded SHP files
- (-) Not directly "land use" but rather population/business density which can proxy land use
- (-) API key required; may have rate limits
- (-) Grid-level business data may require application process
- **Verdict:** STRONG alternative. Business establishment density by sector effectively classifies areas into commercial, industrial, residential without needing explicit zoning data. Population density distinguishes urban from rural. The 100m grid is well-suited to trip-origin matching.

### 1E. VWorld (브이월드, vworld.kr) -- Spatial Information Open Platform

**Source:** [vworld.kr](https://www.vworld.kr/v4po_main.do)

- **Provider:** MOLIT / Korea Land and Geospatial Information Corporation (한국국토정보공사, LX)
- **Format:** WMS/WFS API (versions 1.0 and 2.0), tile services
- **Data available:**
  - Urban planning data (도시계획) -- 15 types including zoning
  - Cadastral maps (연속지적도)
  - Building data
  - LX Map layers
- **Resolution:** Parcel-level
- **Coverage:** Nationwide
- **Availability:** Free API key registration required; for Korean nationals/organizations (may have restrictions for foreign servers)
- **Note:** Absorbed the former 국가공간정보포털 (nsdi.go.kr) as of January 2024

**Assessment:**
- (+) Official zoning data accessible via WFS API -- can query 용도지역 per location
- (+) Programmatic access via Python (OWSLib, requests)
- (+) High-quality cadastral and planning data
- (-) API access may be restricted -- Korean law prohibits export of certain spatial data to foreign servers
- (-) Rate limits likely for 19.5M queries
- (-) Registration process is in Korean; may require Korean phone number/organization
- (-) WFS queries for millions of points would be extremely slow
- **Verdict:** Technically has the right data but impractical for bulk analysis at our scale (19.5M trip origins). Better suited for validation/spot-checking.

### 1F. Building Register API (건축물대장, data.go.kr)

**Source:** [data.go.kr Dataset 15134735](https://www.data.go.kr/data/15134735/openapi.do)

- **Provider:** MOLIT (via 건축HUB)
- **Format:** OpenAPI (XML/JSON responses)
- **Data available:** Building use type (주용도코드/주용도코드명), structure, floor count, area
- **Python library:** [PublicDataReader](https://github.com/WooilJeong/PublicDataReader) (`pip install PublicDataReader`)
- **Coverage:** Nationwide -- all registered buildings
- **Availability:** Free API key from data.go.kr

**Assessment:**
- (+) Building-level use type classification (residential, commercial, office, industrial, etc.)
- (+) Well-documented Python library (PublicDataReader) for easy access
- (+) Nationwide coverage
- (-) Per-building query via API -- impractical for 19.5M trip origins
- (-) No spatial polygons -- buildings identified by address/PNU, not lat/lon
- (-) Would need geocoding step to match buildings to GPS coordinates
- **Verdict:** Useful data but API-based individual queries are infeasible at scale. Could be used if pre-downloaded building data with coordinates is available.

### 1G. LH (한국토지주택공사)

**Source:** [lh.or.kr](https://www.lh.or.kr/menu.es?mid=a10110020000)

- **Provider:** Korea Land and Housing Corporation
- **Format:** Various via public data portal
- **Data available:** Housing complex information, new town development data, COMPAS platform (urban analytics)
- **Coverage:** LH development areas only (not nationwide)

**Assessment:**
- (-) Covers LH-developed areas only, not all 52 cities
- (-) Not a general-purpose land use dataset
- **Verdict:** Not suitable for this study.

---

## 2. OpenStreetMap Landuse Tags in Korea

### 2A. Coverage Assessment (Empirical Tests on Our Data)

**Key finding:** OSM landuse coverage in South Korea is **poor to moderate**.

**Empirical results from our study area (tested 2026-03-13):**

| City | Landuse features | Area coverage | Trip origin match rate |
|------|-----------------|--------------|----------------------|
| Seoul (trip bbox) | 4,778 | 81.6 / 354.6 km² = 23.0% | 14.5% (of 50K sample) |
| Daejeon (trip bbox) | 4,636 | 91.3 / 1,012.8 km² = 9.0% | Not tested |

**Landuse tag distribution (Seoul):** residential (2,022), grass (1,192), commercial (322), forest (287), construction (216), retail (180)

**Building footprint coverage (Seoul, small bbox):** 18,842 buildings — excellent coverage. 99.6% of trip origins have ≥1 building within 200m. BUT 68.3% of buildings are tagged "yes" (unclassified building type) — building type classification is unreliable.

**POI/amenity coverage (Seoul, small bbox):** 4,914 amenities — excellent. Top: restaurant (1,860), parking (630), cafe (537), bicycle_rental (208).

**Conclusion from empirical tests:** OSM landuse polygons match only ~14.5% of trip origins in Seoul (best-case city). This is **insufficient for a control variable**. Building density is excellent but building TYPE is mostly unclassified. POI density provides the best OSM-based land use proxy.

- **Buildings:** A small fraction of Korean buildings are mapped in OSM (estimated under 10% as of 2024, based on community reports; no authoritative source available). Korean law prohibits export of MOLIT building data to foreign servers, which prevents bulk imports into OSM.
- **Landuse polygons:** No specific coverage statistics available, but given the low building mapping rate, landuse polygon coverage is likely to be similarly incomplete.
- **Road networks:** Better covered than buildings/landuse, especially after the Pokemon Go-driven mapping spike (17x increase in daily contributions in 2017).
- **Urban vs rural:** Major city centers (Seoul, Busan) have better coverage; smaller cities and rural areas are likely very sparse.
- **Existing project experience:** The project already uses OSM road network data via osmnx (50 city GeoPackages in `data_parquet/osm_networks/`). We observed that `maxspeed` tag coverage was only 1-12% in Korea -- landuse tags are likely similarly low.

### 2B. Available Tags

| Tag | Description | Expected Korea coverage |
|-----|-------------|------------------------|
| `landuse=residential` | Residential areas | Moderate in major cities, sparse elsewhere |
| `landuse=commercial` | Commercial/office areas | Low -- often not distinguished |
| `landuse=industrial` | Industrial zones | Moderate -- large zones often mapped |
| `landuse=retail` | Shopping areas | Very low |
| `landuse=recreation_ground` | Recreation | Low |
| `amenity=*` (POIs) | Points of interest | Moderate -- restaurants, shops, etc. |

### 2C. Extraction Methods

1. **osmnx `features_from_point()`:** Query landuse tags within buffer around each trip origin. Already installed (v2.0.7).
   ```python
   import osmnx as ox
   tags = {'landuse': True}
   gdf = ox.features_from_point((lat, lon), tags=tags, dist=200)
   ```
   Problem: Overpass API rate limits make this infeasible for 19.5M queries.

2. **Geofabrik PBF + pyrosm:** Download full South Korea PBF (259 MB) and extract landuse polygons locally.
   ```python
   from pyrosm import OSM
   osm = OSM("south-korea-latest.osm.pbf")
   landuse = osm.get_landuse()  # Returns GeoDataFrame
   ```
   Then spatial join with trip origins.

3. **Geofabrik Shapefile:** Pre-extracted landuse layer in GeoPackage (526 MB). Can load directly with geopandas.

**Assessment:**
- (+) Free, no registration needed
- (+) Python-friendly (pyrosm, osmnx, geopandas)
- (+) Already have osmnx infrastructure in the project
- (-) **Coverage is the critical weakness** -- likely missing landuse polygons for many of the 52 cities
- (-) No guarantee of consistent classification across cities
- (-) Korean mappers may use different tagging conventions
- **Verdict:** Feasible to extract but likely too incomplete for a reliable land use variable. Would need to validate coverage city-by-city before relying on it. Best used as a supplementary/validation source, not the primary land use variable.

---

## 3. Alternative Approaches

### 3A. POI Density from OSM or Kakao Map API

#### OSM POI Density
- Extract `amenity`, `shop`, `tourism`, `leisure` tags within buffer of trip origin
- Compute density metrics: commercial POI count, restaurant density, etc.
- Same coverage concerns as landuse tags (estimated under 10% building mapping rate)

#### Kakao Map API
- **Source:** [developers.kakao.com](https://developers.kakao.com/docs/latest/en/local/dev-guide)
- Keyword/category search API with lat/lon + radius queries
- Category codes for restaurants, shops, hospitals, schools, etc.
- Free tier: likely 30,000 requests/day (standard for Korean APIs)
- Problem: 19.5M trip origins / 30,000 per day = 650 days of queries (infeasible without caching/aggregation)
- **Feasible if:** pre-aggregate to H3 hexagons (8,144 unique origins at H3-8) then query POI density per hex. 8,144 queries = 1 day.

**Assessment:**
- Kakao POI density via H3 aggregation is **practically feasible**
- Commercial POI density effectively proxies commercial land use
- Requires Kakao developer account (free)

### 3B. Population Density Grids

#### WorldPop
- **Source:** [portal.worldpop.org](https://www.portal.worldpop.org/)
- 100m x 100m population count grid, GeoTIFF format
- Available for Korea (KOR), 2015-2020 epochs
- Free download, CC-BY license
- Process with rasterio + geopandas for point-in-raster lookup

#### GHS-POP (European Commission GHSL)
- **Source:** [human-settlement.emergency.copernicus.eu](https://human-settlement.emergency.copernicus.eu/ghs_pop.php)
- 100m population grid, GeoTIFF
- 1975-2020 (5-year intervals) + 2025/2030 projections
- Free, open access
- Derived from census + satellite imagery

#### GHS-SMOD (Settlement Model Grid)
- **Source:** [human-settlement.emergency.copernicus.eu](https://human-settlement.emergency.copernicus.eu/ghs_smod2023.php)
- 1km grid classifying cells as: Urban Centre, Dense Urban Cluster, Semi-dense Urban Cluster, Suburban/Peri-urban, Rural Cluster, Low-density Rural, Very Low-density Rural
- Free, open access, GeoTIFF + GPKG
- **Directly provides urban-rural classification at 1km**

**Assessment:**
- (+) Global, consistent, well-validated datasets
- (+) Easy to integrate (GeoTIFF + rasterio)
- (+) Population density is a strong built environment proxy
- (-) 100m-1km resolution may be too coarse for within-city variation
- (-) These are urbanization/density proxies, not land USE classification
- (-) Cannot distinguish commercial from residential -- both have high population
- **Verdict:** Useful as control variables (population density, urban/rural) but insufficient alone for land use classification. GHS-SMOD at 1km is too coarse. WorldPop at 100m is better but only provides population count, not use type.

### 3C. Night Light Imagery (VIIRS)

- **Source:** [eogdata.mines.edu](https://eogdata.mines.edu/products/vnl/) (VIIRS Nighttime Lights)
- Monthly/annual composites, ~500m resolution
- Cloud-optimized GeoTIFF, free download
- Higher radiance = more commercial/industrial activity

**Assessment:**
- (+) Global, consistent, free
- (+) Good proxy for economic activity intensity
- (-) 500m resolution is too coarse for trip-origin-level analysis
- (-) Cannot distinguish land use TYPE (commercial high light = industrial high light)
- (-) Temporal mismatch (nighttime only)
- **Verdict:** Too coarse and not discriminative enough for land use classification. Not recommended.

### 3D. Building Footprint Density (OSM or Microsoft)

- OSM building footprints: estimated under 10% coverage in Korea (no authoritative source) -- too incomplete
- Microsoft Building Footprints: provides ML-detected building footprints globally, but Korea may have lower quality due to dense urban fabric
- Building density alone does not distinguish residential from commercial

**Assessment:** Not recommended as primary variable.

---

## 4. Practical Considerations Summary

### 4A. Comparison Matrix

| Source | Land Use Type? | Resolution | Coverage (52 cities) | Format | Effort | Feasibility |
|--------|---------------|------------|---------------------|--------|--------|-------------|
| **MOLIT 토지이용현황도** | Land COVER only (urban/forest) | Parcel | All 52 | SHP | Low | LOW (wrong classification) |
| **MOLIT 토지이용계획정보** | Zoning (residential/commercial/industrial) | Parcel | All 52 | CSV+SHP | Medium-High | HIGH (best match) |
| **VWorld WFS API** | Zoning | Parcel | All 52 | API | High | LOW (rate limits) |
| **SGIS Business Census** | Business density by sector | 100m grid | All 52 | API/SHP | Medium | HIGH (good proxy) |
| **SGIS Population Grid** | Population density | 100m grid | All 52 | API/SHP | Medium | HIGH (good control) |
| **OSM landuse tags** | landuse=* | Polygon | Partial (major cities) | PBF/SHP | Low | LOW (incomplete) |
| **OSM POI density** | amenity/shop counts | Point | Partial | PBF/SHP | Low | MEDIUM |
| **Kakao POI density** | Category counts | Point | All 52 | API | Medium | MEDIUM-HIGH (via H3) |
| **WorldPop** | Population count | 100m | All 52 | GeoTIFF | Low | HIGH (easy) |
| **GHS-SMOD** | Urban/rural class | 1km | All 52 | GeoTIFF | Low | HIGH (easy, coarse) |
| **VIIRS Night Light** | Radiance | 500m | All 52 | GeoTIFF | Low | LOW (too coarse) |
| **Building Register API** | Building use type | Building | All 52 | API | Very High | LOW (scale issue) |

### 4B. Decision Matrix (PI Quick Reference)

The following matrix summarizes the key decision factors for each viable data source. Sources are qualitatively tiered based on combined coverage, granularity, effort, cost, and update frequency.

| Source | Coverage | Granularity | Effort | Cost | Update | Tier |
|--------|----------|-------------|--------|------|--------|------|
| **MOLIT 토지이용계획정보** | All 52 | Parcel zoning | 2-3 days | Free (registration) | Annual | **Recommended** |
| **Kakao POI + WorldPop** | All 52 | H3 hex + 100m | 1-2 days | Free tier | Continuous | **Recommended** |
| **WorldPop only** | All 52 | 100m density | 30 min | Free | Annual | **Recommended** (min. viable) |
| **SGIS Business Census** | All 52 | 100m grid | 2-3 days | Free (application) | 5-year | Viable |
| **GHS-SMOD** | All 52 | 1km class | 30 min | Free | 5-year | Viable (coarse) |
| **VWorld WFS** | All 52 | Parcel zoning | Rate-limited | Free (restricted) | Real-time | Viable (API barriers) |
| **OSM landuse** | Major only | Polygon | 1 day | Free | Irregular | Not recommended (14.5% match) |
| **Building Register** | All 52 | Building-level | Infeasible | Free | Real-time | Not recommended (scale) |
| **VIIRS Night Light** | All 52 | 500m | 30 min | Free | Monthly | Not recommended (too coarse) |

**Top 3 recommendations for PI decision:**
1. **MOLIT zoning** — Gold standard classification, moderate effort. Best if precision matters.
2. **Kakao POI + WorldPop** — Practical hybrid, well-established in literature. Best effort/value ratio.
3. **WorldPop only** — Minimum viable (30 min). Best if time-constrained; state as limitation.

### 4C. Recommended Approach (Ranked)

#### Option 1 (Best): MOLIT 토지이용계획정보 (Zoning Data)
- Download parcel-level zoning classification from data.go.kr (dataset 15045900)
- Obtain cadastral boundary polygons (연속지적도) from VWorld or data.go.kr to create spatial polygons
- Spatial join: for each trip origin (lat/lon), find enclosing parcel polygon, get zoning class
- Reclassify into study categories: Residential, Commercial, Industrial, Green, Mixed, Other
- **Pros:** Exact zoning classification, parcel-level, official data
- **Cons:** Requires downloading/processing large cadastral boundary files; joining two datasets (zoning + boundaries)
- **Estimated effort:** 2-3 days

#### Option 2 (Practical): SGIS Business + Population Census Grids
- Download business establishment counts by industry sector at 100m grid level from SGIS
- Download population counts at same grid level
- For each trip origin, look up the containing grid cell
- Compute derived variables:
  - `pop_density`: population per grid cell (residential proxy)
  - `commercial_density`: retail/food/service establishments per grid cell
  - `industrial_density`: manufacturing/warehouse establishments per grid cell
  - `mixed_use_index`: entropy or ratio of commercial/residential/industrial
- **Pros:** 100m resolution is excellent; business census is comprehensive; well-suited to Python workflow
- **Cons:** Proxy rather than official zoning; requires SGIS API registration; data application process
- **Estimated effort:** 2-3 days

#### Option 3 (Quickest): WorldPop + GHS-SMOD + OSM Landuse
- Download WorldPop 100m population grid for Korea (free GeoTIFF)
- Download GHS-SMOD 1km settlement classification (free GeoTIFF)
- Extract OSM landuse polygons from Geofabrik PBF (free, local processing)
- For each trip origin: population density from WorldPop, urban class from GHS-SMOD, landuse from OSM (where available)
- **Pros:** All free, no registration, quick integration
- **Cons:** OSM landuse is incomplete; GHS-SMOD is 1km (coarse); population density alone does not distinguish land use type
- **Estimated effort:** 1 day

#### Option 4 (Hybrid -- Recommended for Paper): Kakao POI via H3 + WorldPop
- Aggregate trip origins to H3 hexagons (already computed: 8,144 unique at H3-8)
- Query Kakao Map API for POI counts by category within each hexagon centroid + radius
- Download WorldPop 100m population grid
- Variables:
  - `poi_commercial`: restaurant + shop + office POI count within 250m
  - `poi_industrial`: factory + warehouse POI count
  - `pop_density`: WorldPop 100m grid value
  - `land_use_mix`: Shannon entropy of POI category proportions
- **Pros:** Complete coverage of all 52 cities; manageable API calls (8K not 19M); well-established in transport literature
- **Cons:** POI density is a proxy; Kakao requires developer account; Shannon entropy requires interpretation
- **Estimated effort:** 1-2 days

### 4D. Integration with Existing Workflow

The study already has:
- Trip origins as `start_lat`, `start_lon` in `trip_modeling.parquet` (19.5M trips)
- H3 hexagon assignment (resolution 8, 8,144 unique hexes)
- OSM road network GeoPackages for 50 cities (`data_parquet/osm_networks/`)
- scipy.spatial.cKDTree infrastructure for nearest-neighbor matching
- geopandas, DuckDB, duckdb for spatial operations

Any chosen approach would follow:
1. Prepare land use layer (polygons, grid, or hexagon-level values)
2. Spatial join or raster lookup for each trip origin
3. Add column(s) to `trip_modeling.parquet`
4. Include as control variable in FE/GEE models

### 4E. Licensing Summary

| Source | License | Attribution Required? | Export Restrictions? |
|--------|---------|----------------------|---------------------|
| MOLIT data.go.kr | 공공누리 Type 1 | Yes | No |
| SGIS | Government open data | Yes | No |
| VWorld | Free but restricted | Yes | **Yes -- export to foreign servers prohibited** |
| OSM | ODbL | Yes (share-alike) | No |
| WorldPop | CC-BY 4.0 | Yes | No |
| GHS-SMOD | CC-BY 4.0 | Yes | No |
| Kakao API | Free tier | Yes | Terms of service apply |

---

## 5. Recommendation for This Paper

**For the AMAR paper, I recommend Option 2 (SGIS Business + Population Census) or Option 4 (Kakao POI via H3 + WorldPop) as the most practical approaches.**

Rationale:
1. **Option 1 (MOLIT zoning)** is the gold standard but requires assembling cadastral boundary polygons + zoning data, which is a non-trivial data engineering task.
2. **Option 2 (SGIS)** provides the most rigorous land use proxy -- business establishment density by sector directly measures the economic function of an area. However, it requires navigating the SGIS data application process.
3. **Option 4 (Kakao POI + WorldPop)** is the most practical for a paper supplement -- 8,144 API calls is feasible, WorldPop is a single GeoTIFF download, and POI density is a well-established built environment measure in the transport literature (commonly used in e-scooter studies).

**Minimum viable approach (if time-constrained):**
- WorldPop 100m population density only (1 GeoTIFF download, ~30 minutes to integrate)
- Add `pop_density_100m` as a continuous control variable
- This controls for urban density without distinguishing land use type
- Limitation is clearly stated in paper

**If collaborators specifically want residential/commercial/industrial classification:**
- Must use MOLIT 토지이용계획정보 (Option 1) or SGIS business census (Option 2)
- OSM landuse is NOT reliable enough for Korea to serve as primary variable

---

## Sources

- [MOLIT 토지이용현황도 (data.go.kr 3049890)](https://www.data.go.kr/data/3049890/fileData.do)
- [MOLIT 토지이용현황도 2024 update (data.go.kr 15059724)](https://www.data.go.kr/data/15059724/fileData.do)
- [MOLIT 토지이용계획정보 (data.go.kr 15045900)](https://www.data.go.kr/data/15045900/fileData.do)
- [토지이음 Land Use Regulation Portal](https://www.eum.go.kr/)
- [SGIS Statistical Geographic Information Service](https://sgis.kostat.go.kr/)
- [SGIS Developer Center](https://sgis.mods.go.kr/developer/html/openApi/api/data.html)
- [VWorld Spatial Information Platform](https://www.vworld.kr/v4po_main.do)
- [VWorld WMS/WFS API Reference](https://vworld.kr/dev/v4dv_wmsguide_s001.do)
- [V-World API Samples (GitHub)](https://github.com/V-world/V-world_API_sample)
- [Building Register API (data.go.kr 15134735)](https://www.data.go.kr/data/15134735/openapi.do)
- [PublicDataReader Python Library](https://github.com/WooilJeong/PublicDataReader)
- [Geofabrik South Korea OSM Download](https://download.geofabrik.de/asia/south-korea.html)
- [OSM Korea Wiki](https://wiki.openstreetmap.org/wiki/Korea)
- [OSM Interview: OpenStreetMap in Korea](https://blog.opencagedata.com/post/openstreetmap-in-korea)
- [Pyrosm Documentation](https://pyrosm.readthedocs.io/en/latest/)
- [OSMnx Documentation](https://osmnx.readthedocs.io/en/stable/user-reference.html)
- [Kakao Local API Documentation](https://developers.kakao.com/docs/latest/en/local/dev-guide)
- [WorldPop Data Portal](https://www.portal.worldpop.org/)
- [GHS-POP (European Commission)](https://human-settlement.emergency.copernicus.eu/ghs_pop.php)
- [GHS-SMOD Settlement Classification](https://human-settlement.emergency.copernicus.eu/ghs_smod2023.php)
- [VIIRS Nighttime Lights](https://eogdata.mines.edu/products/vnl/)
- [OSM Landuse Coverage Study (ScienceDirect)](https://www.sciencedirect.com/science/article/abs/pii/S0143622822001138)
- [Korean Planning Law (ResearchGate)](https://www.researchgate.net/publication/376467937_Planning_Law_and_Development_Process_in_South_Korea)
- [Korean Zoning System (ResearchGate)](https://www.researchgate.net/figure/Land-use-zoning-system-of-the-urban-area-in-Korea_tbl1_348245386)
- [E-scooter Built Environment Study (ScienceDirect)](https://www.sciencedirect.com/science/article/abs/pii/S096669232100137X)
