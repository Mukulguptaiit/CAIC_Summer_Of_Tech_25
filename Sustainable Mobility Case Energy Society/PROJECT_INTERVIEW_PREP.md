# Low-Emission Equitable Urban Mobility Plan - Interview Prep

**Duration:** May 2025 - July 2025  
**Organization:** Energy Society (Mobility Case 2025)

---

## 🎯 Project Overview (30-second pitch)
*"I designed a comprehensive urban mobility plan for a metropolitan city that reduces transport emissions by 38% while improving accessibility for underserved communities. Using multi-objective optimization, I balanced competing goals of emission reduction, social equity, and travel time - optimizing public transit routes, EV infrastructure placement, and modal shift strategies with data-driven analysis and simulation."*

---

## 📊 Technical Implementation

### Part 1: Baseline Analysis & Problem Identification

**City Context:** Metropolitan area with 5M population, 50 transport corridors, 20 zones

**Data Analysis:**

**1. Emission Calculation (Bottom-Up Approach)**
```
Emissions = Σ (Vehicle_km × Emission_Factor × Modal_Share)

Emission Factors:
- Private car (ICE): 120 g CO2/km
- Bus (diesel): 30 g CO2/passenger-km  
- Metro: 20 g CO2/passenger-km
- Two-wheeler: 50 g CO2/km
- Bicycle/Walking: 0 g CO2/km
```

**Baseline Results:**
- Total annual emissions: 2.8 million tons CO2
- 68% from private vehicles (modal share: 52%)
- 22% from two-wheelers (modal share: 28%)
- 10% from public transit (modal share: 20%)

**Interview Questions:**

**Q: How did you calculate emissions?**
*A: Bottom-up approach using VKT (vehicle kilometers traveled) × emission factors from IPCC and TERI databases. For each mode, multiplied daily trips × average trip distance × emission factor × 365 days. Considered load factors for buses (35 passengers/bus) and occupancy for cars (1.2 persons/car).*

**Q: What about electric vehicles?**
*A: EV emissions depend on electricity grid carbon intensity. India average: 630 g CO2/kWh. EV efficiency: 0.2 kWh/km → 126 g CO2/km well-to-wheel (similar to ICE!). However, grid is decarbonizing (50% renewable by 2030 target), so EVs become cleaner over time. Also included upstream emissions from battery manufacturing.*

**2. Accessibility Analysis**

**Methodology:**
- Generated 15-minute walking isochrones around metro/bus stops using network distance
- Overlaid with population density data
- Calculated percentage of population within service area

**Results:**
- Only 42% of population within 15-min of quality transit
- Low-income neighborhoods: 28% coverage
- High-income neighborhoods: 67% coverage
- **Equity gap identified**

**Interview Questions:**

**Q: Why 15-minute metric?**
*A: Based on "15-Minute City" concept - residents should access daily needs within 15-min walk/cycle. Research shows people willing to walk 10-15 min to transit. Beyond this, they switch to private vehicles. Standard in urban planning (Paris, Melbourne, Portland).*

**Q: Isochrone vs Euclidean distance?**
*A: Isochrone uses actual road network (accounts for barriers like rivers, highways), gives realistic reachability. Euclidean (straight-line) overestimates access - crow flies but people can't. Used NetworkX with OSM road data for network routing.*

**3. High-Emission Corridor Identification**

**Criteria:**
- CO2 tons/day > 75th percentile (threshold: 850 tons)
- Low public transit coverage (<30%)
- High vehicle volume (>40,000 vehicles/day)

**Top 5 High-Emission Corridors:**
| Corridor | Daily CO2 (tons) | Transit Coverage | Income Level | Priority |
|----------|-----------------|------------------|--------------|----------|
| Ring Road North | 1,240 | 18% | Low | Critical |
| Highway 8 | 1,150 | 22% | Mixed | High |
| MG Road | 980 | 35% | High | Medium |
| Airport Link | 920 | 15% | Low | Critical |
| Industrial Zone | 890 | 12% | Low | Critical |

**Interview Question:**

**Q: How did you prioritize interventions?**
*A: Multi-criteria scoring: 40% emission reduction potential, 30% equity (income level + current access), 30% feasibility (political will, land availability, budget). Critical corridors score high on all three. This ensures solutions are both effective and equitable.*

---

### Part 2: Multi-Objective Optimization

**Objective Function:**
```
Maximize: 0.4 × Emission_Reduction + 0.3 × Equity_Score + 0.3 × Time_Savings
Subject to:
  - Budget ≤ $50 million
  - Σ weights = 1
  - Intervention constraints (bus lanes only on roads >30m width)
```

**Decision Variables:**
- Binary: transit_upgrade[i] for corridor i (0 or 1)
- Binary: bike_lane[i] for corridor i (0 or 1)

**Optimization Method:** Linear Programming (PuLP)

**Interview Questions:**

**Q: Why linear programming for multi-objective?**
*A: Converted to single objective using weighted sum (scalarization). Weights represent stakeholder priorities - calibrated through surveys and policy goals. Alternative: ε-constraint method (optimize one objective, constrain others) or Pareto frontier (show trade-off curve).*

**Q: How do you set weights (0.4, 0.3, 0.3)?**
*A: Based on: 1) City's climate commitments (40% emission reduction by 2030), 2) Social equity mandate (constitutional), 3) Economic productivity (time is GDP). Validated with sensitivity analysis - solution robust to ±10% weight changes.*

**Q: Why binary variables instead of continuous?**
*A: You either build a bus lane or you don't - no fractional interventions. Makes it Mixed Integer Linear Programming (MILP), computationally harder but realistic. Solved using branch-and-bound.*

**Optimal Solution:**
- 12 corridors get transit upgrades (bus priority lanes)
- 18 corridors get protected bike lanes
- Total cost: $48.2M (within budget)

**Expected Impact:**
- Emission reduction: 38% over baseline
- Population with 15-min access: 42% → 71% (+29 points)
- Average commute time: 47 min → 38 min (-19%)

---

### Part 3: EV Infrastructure Optimization

**Problem:** Place 50 EV charging stations to maximize coverage and demand satisfaction

**Formulation:**
```
Maximize: Σ (EV_adoption_potential_i × coverage_ij)
Subject to:
  - Σ station_j = 50 (exactly 50 stations)
  - coverage_ij = 1 if distance_ij < 5km, else 0
  - Each zone covered by ≥1 station
  - Budget constraint
```

**Methodology:**
- Calculated EV adoption potential per zone: f(income, car_ownership, charging_access)
- Used greedy + local search heuristic (p-median variant)
- Validated with coverage distance matrix

**Results:**
- 50 stations cover 89% of potential EV owners within 5km
- High-density zones get 2-3 stations (fast chargers)
- Park-and-ride locations prioritized
- Estimated 15,000 EVs supported by 2030

**Interview Questions:**

**Q: Why 5km coverage radius?**
*A: Based on research: EV owners charge at home (70%), workplace (15%), public (15%). Public chargers for opportunistic charging - 5km ensures convenience without detours. Also, most urban trips <10km so 5km is half typical trip.*

**Q: Fast chargers vs slow chargers?**
*A: Fast (50kW DC): 80% charge in 30 min, expensive ($100k/unit), highway/commercial areas. Slow (7kW AC): Full charge overnight, cheap ($5k/unit), residential areas. Optimized mix: 20 fast, 30 slow based on land use and dwelling type.*

**Q: How did you project EV adoption?**
*A: Logistic growth model: S-curve with 70% maximum penetration by 2040. Growth rate calibrated from Norway (leader: 90% new cars electric). Policy sensitivity: FAME-II subsidies, charging infrastructure, fuel price. Conservative estimate: 25% by 2030.*

**EV Adoption Projection:**
| Year | EV Share | Total EVs | Chargers Needed | Investment |
|------|----------|-----------|-----------------|------------|
| 2025 | 5% | 5,000 | 10 | $0.5M |
| 2030 | 25% | 25,000 | 50 | $2.5M |
| 2035 | 50% | 50,000 | 120 | $6.0M |
| 2040 | 70% | 70,000 | 180 | $9.0M |

---

### Part 4: Public Transit Route Optimization

**Objective:** Design 5 new bus routes to maximize ridership while respecting budget

**Constraints:**
- Each route serves 3-15 zones (connectivity)
- Route operational cost proportional to length
- Capacity constraints (max 200 passengers/bus-hour)
- Prioritize underserved areas

**Algorithm:** Greedy heuristic + local search
1. Start with zones of highest unmet demand
2. Add adjacent zones maximizing ridership/cost ratio
3. Ensure connectivity (no isolated zones)
4. Local search: swap zones to improve objective

**Optimized Routes:**
| Route | Zones Served | Length (km) | Est. Ridership/day | Cost/day |
|-------|--------------|-------------|-------------------|----------|
| R1 | 8 | 24 | 12,500 | $3,200 |
| R2 | 12 | 35 | 18,200 | $4,500 |
| R3 | 6 | 18 | 8,900 | $2,400 |
| R4 | 10 | 28 | 14,300 | $3,600 |
| R5 | 7 | 22 | 10,100 | $2,900 |

**Total:** 43 zones served, 64,000 riders/day, $16,600 operating cost/day

**Interview Questions:**

**Q: How did you estimate ridership?**
*A: Gravity model: Trips_ij = k × (Pop_i × Jobs_j) / Distance_ij². Parameters k calibrated from existing routes. Also considered: income (elasticity = -0.4), fare, travel time, current modal share. Validated against actual ridership data (R² = 0.83).*

**Q: Why not use exact algorithms for route design?**
*A: Vehicle Routing Problem (VRP) is NP-hard. Exact solution only feasible for <20 zones. We have 43 zones → 10^50 possible routes. Heuristics give 85-95% optimal solution in minutes vs days for exact. Good enough for planning - refine later with detailed analysis.*

**Q: How to determine bus frequency?**
*A: Demand-based: High demand (>10k/day) → 6 buses/hour (10-min headway). Medium (5-10k) → 4 buses/hour (15-min). Low (<5k) → 2 buses/hour (30-min). Constraint: Total fleet = 120 buses. Optimized frequency allocation using LP to maximize ridership within fleet constraint.*

---

### Part 5: Modal Shift Simulation

**Scenarios Developed:**

**Baseline (Business as Usual):**
- No interventions
- Current trends continue
- 2% annual increase in private vehicle ownership

**Scenario 1 (Moderate):**
- 12 transit corridors improved
- 18 bike lanes added
- 50 EV chargers
- 20% annual EV growth

**Scenario 2 (Ambitious):**
- All scenario 1 + congestion pricing downtown
- Metro line extension
- 30% annual EV growth
- Car-free zones in commercial areas

**Simulation Model:**
```python
Modal_Share_t = Modal_Share_0 × e^(intervention_effect × t)

Effects calibrated from literature:
- Bus lane → +8% bus ridership (elasticity from Bogotá BRT)
- Bike lane → +12% cycling (Copenhagen data)
- EV subsidy → +15% EV adoption (Norway)
- Congestion pricing → -18% car trips (London)
```

**10-Year Projections:**

| Metric | Baseline | Scenario 1 | Scenario 2 |
|--------|----------|------------|------------|
| Private Car | 52% → 58% | 52% → 38% | 52% → 28% |
| Public Transit | 20% → 18% | 20% → 34% | 20% → 42% |
| Cycling/Walking | 8% → 7% | 8% → 16% | 8% → 22% |
| Two-wheeler | 20% → 17% | 20% → 12% | 20% → 8% |
| **Emissions (Index)** | **100 → 118** | **100 → 62** | **100 → 45** |
| **Avg Commute (min)** | **47 → 52** | **47 → 40** | **47 → 35** |

**Interview Questions:**

**Q: How did you validate the simulation?**
*A: 1) Calibrated parameters from peer-reviewed studies, 2) Backtested on cities with similar interventions (Bogotá, Curitiba), 3) Sensitivity analysis (±20% on elasticities), 4) Expert review from transport planners, 5) Sanity checks (modal shares sum to 100%, emissions decrease monotonically).*

**Q: What assumptions could break the model?**
*A: 1) COVID-style disruptions (work-from-home), 2) Autonomous vehicles (unpredictable impact), 3) Fuel price shocks, 4) Political changes (policies reversed), 5) Behavioral factors (cultural resistance to cycling), 6) Budget cuts, 7) Technology leaps (hyperloop).*

**Q: How do you handle uncertainty in 10-year projections?**
*A: Monte Carlo simulation with 1,000 runs. Varied key parameters (EV adoption rate, bus ridership elasticity, implementation delays) based on probability distributions. Report 5th-95th percentile range. Scenario 1 emissions reduction: 35-42% (95% CI). Also scenario analysis (optimistic/pessimistic) for stakeholder communication.*

---

### Part 6: Visualization & Communication

**Tools Used:**
- **QGIS:** Spatial analysis, isochrone mapping, corridor overlays
- **Matplotlib:** Time-series projections, modal share evolution, emission trends
- **Seaborn:** Heatmaps (accessibility by zone), correlation matrices
- **Plotly:** Interactive dashboards for stakeholder presentations

**Key Visualizations Created:**

1. **Accessibility Heatmap** (Before/After)
   - Color-coded zones by 15-min transit access
   - Clearly shows equity improvements

2. **Emission Corridor Map**
   - Width = vehicle volume, Color = CO2 intensity
   - Overlayed with proposed interventions

3. **Modal Share Evolution** (Line plot)
   - 3 scenarios over 10 years
   - Stacked area chart showing mode composition

4. **Pareto Frontier** (Scatter plot)
   - Trade-off between cost and emission reduction
   - Identifies efficient solutions

**Interview Question:**

**Q: How do you communicate technical results to non-technical stakeholders?**
*A: 1) Start with problem (visual: congestion, pollution), 2) Simple metrics (38% emission cut, 15-min city), 3) Maps and charts (avoid equations), 4) Compare scenarios (baseline vs intervention), 5) Highlight co-benefits (health, equity, jobs), 6) Use analogies ("equivalent to removing 50,000 cars"), 7) Interactive dashboards for exploration.*

---

## 🔧 Technical Stack

**Libraries:**
- **Optimization:** PuLP (linear programming), SciPy (multi-objective)
- **Spatial Analysis:** GeoPandas, Shapely, NetworkX (routing)
- **Data Processing:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn, Plotly, QGIS
- **Simulation:** Custom Python (modal shift dynamics)

---

## 💡 Key Learnings & Challenges

**Challenge 1: Data scarcity**
- **Problem:** City didn't have origin-destination (O-D) matrix
- **Solution:** Estimated using gravity model + household travel survey (small sample)
- **Learning:** Work with imperfect data, validate with proxies (Google traffic, bus ridership)

**Challenge 2: Conflicting objectives**
- **Problem:** Cheapest solution ignored equity, greenest solution too expensive
- **Solution:** Multi-objective optimization, stakeholder workshops to set weights
- **Learning:** No single "best" solution - facilitate informed trade-offs

**Challenge 3: Political constraints**
- **Problem:** Car-free zones politically infeasible in business districts
- **Solution:** Proposed pilot in 1 street, monitored for 6 months, then expand
- **Learning:** Incrementalism, build trust with data

**Challenge 4: Behavioral assumptions**
- **Problem:** People don't always act rationally (won't cycle even with perfect infrastructure)
- **Solution:** Included behavioral nudges (gamification, social norms), conservative adoption rates
- **Learning:** Technology + policy insufficient without behavior change

---

## 📈 Business & Social Impact

**Environmental Impact:**
- 38% CO2 reduction = 1.06 million tons/year saved
- Equivalent to planting 17 million trees
- Meets city's 2030 climate target

**Social Equity:**
- Low-income transit access: 28% → 73%
- Reduced transport cost burden from 24% to 15% of income
- 45,000 jobs created (construction, operations)

**Economic Impact:**
- Travel time savings: 9 min/trip × 5M trips/day = $180M/year productivity gain
- Health benefits (reduced air pollution): $250M/year (WHO value of statistical life)
- ROI: Benefit-cost ratio of 3.2:1 over 20 years

**Interview Question:**

**Q: How do you calculate economic benefits of transport improvements?**
*A: 1) Time savings (wage rate × hours saved × trip volume), 2) Vehicle operating cost savings (fuel, maintenance), 3) Accident reduction (safety value), 4) Health (air quality → reduced respiratory diseases, active transport → reduced obesity), 5) Property value increase (hedonic pricing). Used transport economics guidelines (US DOT, UK DfT). Discounted at 4% real rate over project lifetime.*

---

## 🎤 Elevator Pitch (1 minute)

*"In this project, I developed a comprehensive low-emission mobility plan for a 5-million person metropolitan area. I started by analyzing baseline emissions and accessibility, identifying high-emission corridors and underserved neighborhoods - finding that only 28% of low-income residents had access to quality transit.*

*Using multi-objective linear programming, I optimized transit route placement, bike lane networks, and EV charging infrastructure to balance emission reduction, social equity, and travel time. The optimal solution allocates a $50M budget across 12 transit corridors and 18 bike lanes.*

*I then simulated modal shift dynamics over 10 years under different scenarios. The ambitious scenario achieves 38% emission reduction while improving transit access from 42% to 71% of the population - demonstrating that environmental and equity goals can be achieved together.*

*Finally, I created interactive visualizations and maps using QGIS and Plotly to communicate findings to city officials, resulting in adoption of the transit plan in the city's 2030 master plan. The project showcases my ability to tackle complex multi-objective problems, integrate spatial analysis, and translate technical work into policy action."*

---

## 🔍 Common Interview Questions

**Q: How is this different from typical optimization problems?**
*A: Three key differences: 1) Multi-objective (emissions + equity + time), not single objective, 2) Spatial constraints (network topology, land use), 3) Behavioral uncertainty (people's mode choices). Required combining optimization, GIS, and simulation - more interdisciplinary than typical OR problems.*

**Q: What's the role of machine learning here?**
*A: Limited - mostly optimization and simulation. Could use ML for: 1) Demand prediction (XGBoost on historical ridership), 2) Image recognition (count vehicles from traffic cameras), 3) NLP (analyze public comments on transit plans). Didn't use because: small data, interpretability needed for policy, optimization more appropriate.*

**Q: How would you implement this in practice?**
*A: Phased approach: 1) Quick wins (bus lanes, low-hanging fruit) Year 1, 2) Infrastructure (bike lanes, chargers) Year 1-3, 3) Behavior change campaigns Year 2-5, 4) Monitoring and adaptive management (adjust routes based on real data) ongoing. Also: Build coalition, secure funding, community engagement, pilot projects.*

**Q: What data would you collect for monitoring?**
*A: 1) Ridership (automated passenger counting on buses), 2) Traffic volumes (loop detectors, cameras), 3) Air quality (PM2.5, NO2 sensors), 4) Travel time (Google Maps API, GPS), 5) Mode share (periodic household surveys), 6) Equity metrics (access by income quintile), 7) Safety (accident reports). Dashboard with quarterly updates to track against targets.*

**Q: Limitations of your approach?**
*A: 1) Assumed linear relationships (diminishing returns in reality), 2) Static network (didn't consider induced demand from new roads), 3) Aggregated zones (within-zone heterogeneity ignored), 4) Perfect implementation (real world has delays, cost overruns), 5) No land use-transport feedback (density changes over time). Future work: agent-based models, integrate with land use, dynamic network equilibrium.*

---

## 📚 References & Further Reading

- Vuchic (2005) - Urban Transit: Operations, Planning, Economics
- Litman (2021) - Evaluating Public Transit Benefits and Costs
- IPCC (2014) - Transport Emissions Calculation Guidelines
- Moreno et al. (2021) - Introducing the "15-Minute City"
- Cervero & Kockelman (1997) - Travel Demand and 3Ds (Density, Diversity, Design)

---

**Pro Tip:** For urban planning/sustainability roles, emphasize equity and stakeholder engagement alongside technical skills. Show you understand that optimal ≠ implementable without community buy-in.
