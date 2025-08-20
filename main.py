"""
Advanced Film Production Schedule Optimizer
Location-First Approach with Graduated Constraint Penalties
Updated to prioritize geographic location clustering
"""

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta, date
import re
import json
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
import hashlib

app = FastAPI(title="Location-First Film Schedule Optimizer v4.0")

# Enable CORS for n8n cloud
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Constants
WORKING_DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]

# Graduated Penalty System
PENALTY_HARD_CONSTRAINT = -10000      # Hard constraint violations
PENALTY_LOCATION_SPLIT = -2000        # Location split across days
PENALTY_DIRECTOR_MANDATE = -5000      # Director mandate violations
PENALTY_TRAVEL_PER_MILE = -100        # Travel time penalty
PENALTY_ACTOR_IDLE_DAY = -200         # Actor idle days
BONUS_SOFT_CONSTRAINT = 100           # Soft constraint satisfaction

class ConstraintPriority(Enum):
    """Constraint hierarchy levels"""
    DIRECTOR = 1
    DOP = 2
    PRODUCTION = 3
    PREP_WRAP = 4
    ACTOR = 5
    TIME_ESTIMATE = 6
    LOCATION = 7
    EQUIPMENT = 8
    WEATHER = 9

class ConstraintType(Enum):
    """Types of constraints"""
    HARD = "hard"
    SOFT = "soft"

@dataclass
class Constraint:
    """Generic constraint representation"""
    source: ConstraintPriority
    type: ConstraintType
    description: str
    affected_scenes: List[str]
    date_restriction: Optional[Dict] = None
    actor_restriction: Optional[Dict] = None
    location_restriction: Optional[Dict] = None

@dataclass
class LocationCluster:
    """Group of scenes at the same geographic location"""
    location: str
    scenes: List[Dict]
    total_hours: float  # RENAMED: Now clearly represents total estimated hours
    estimated_days: int
    required_actors: List[str]

class ScheduleRequest(BaseModel):
    """Request model for structured constraints from n8n"""
    stripboard: List[Dict[str, Any]]
    constraints: Dict[str, Any]
    ga_params: Optional[Dict[str, Any]] = {
        "phase1_population": 50,
        "phase1_generations": 200,
        "phase2_population": 100,
        "phase2_generations": 500,
        "mutation_rate": 0.02,
        "crossover_rate": 0.85,
        "seed": 42,
        "conflict_tolerance": 0.1
    }

class ScheduleResponse(BaseModel):
    """Response model with optimized schedule"""
    schedule: List[Dict[str, Any]]
    summary: Dict[str, Any]              # ADD THIS LINE
    missing_scenes: Dict[str, Any]       # ADD THIS LINE  
    conflicts: List[Dict[str, Any]]
    metrics: Dict[str, Any]
    fitness_score: float
    processing_time_seconds: float

class StructuredConstraintParser:
    """Parses structured constraints from n8n AI agents"""
    
    def __init__(self):
        pass
    
    def parse_all_constraints(self, constraints_dict: Dict[str, Any]) -> List[Constraint]:
        """Parse all structured constraint groups from n8n"""
        all_constraints = []
        
        print(f"DEBUG: Starting constraint parsing with keys: {list(constraints_dict.keys())}")
        
        try:
            if 'people_constraints' in constraints_dict:
                print("DEBUG: Parsing people_constraints")
                people_constraints = self._parse_people_constraints(constraints_dict['people_constraints'])
                all_constraints.extend(people_constraints)
                print(f"DEBUG: Added {len(people_constraints)} people constraints")
            
            if 'location_constraints' in constraints_dict:
                print("DEBUG: Parsing location_constraints")
                location_constraints = self._parse_location_constraints(constraints_dict['location_constraints'])
                all_constraints.extend(location_constraints)
                print(f"DEBUG: Added {len(location_constraints)} location constraints")
            
            if 'technical_constraints' in constraints_dict:
                print("DEBUG: Parsing technical_constraints")
                technical_constraints = self._parse_technical_constraints(constraints_dict['technical_constraints'])
                all_constraints.extend(technical_constraints)
                print(f"DEBUG: Added {len(technical_constraints)} technical constraints")
            
            if 'creative_constraints' in constraints_dict:
                print("DEBUG: Parsing creative_constraints")
                creative_constraints = self._parse_creative_constraints(constraints_dict['creative_constraints'])
                all_constraints.extend(creative_constraints)
                print(f"DEBUG: Added {len(creative_constraints)} creative constraints")
            
            if 'operational_data' in constraints_dict:
                print("DEBUG: Parsing operational_data")
                operational_constraints = self._parse_operational_data(constraints_dict['operational_data'])
                all_constraints.extend(operational_constraints)
                print(f"DEBUG: Added {len(operational_constraints)} operational constraints")
        
        except Exception as e:
            print(f"DEBUG: Error in constraint parsing: {e}")
            import traceback
            traceback.print_exc()
        
        print(f"DEBUG: Total constraints parsed: {len(all_constraints)}")
        return all_constraints
    
    def _parse_people_constraints(self, people_data: Dict) -> List[Constraint]:
        """Parse actor availability constraints - WITH CAST MAPPING EXTRACTION"""
        constraints = []
        
        try:
            if 'actors' in people_data:
                actors_data = people_data['actors']
                
                if isinstance(actors_data, dict):
                    for actor_name, actor_info in actors_data.items():
                        if not isinstance(actor_info, dict):
                            print(f"DEBUG: Skipping {actor_name}, not a dict: {type(actor_info)}")
                            continue
                        
                        constraint_level = actor_info.get('constraint_level', 'Hard')
                        constraint_type = ConstraintType.HARD if constraint_level == 'Hard' else ConstraintType.SOFT
                        
                        # Parse unavailable dates
                        if actor_info.get('dates'):
                            for date_str in actor_info['dates']:
                                constraints.append(Constraint(
                                    source=ConstraintPriority.ACTOR,
                                    type=constraint_type,
                                    description=f"{actor_name} unavailable on {date_str}",
                                    affected_scenes=[],
                                    actor_restriction={
                                        'actor': actor_name, 
                                        'unavailable_date': date_str
                                    }
                                ))
                        
                        # Parse week restrictions
                        if actor_info.get('weeks'):
                            constraints.append(Constraint(
                                source=ConstraintPriority.ACTOR,
                                type=constraint_type,
                                description=f"{actor_name} available weeks: {actor_info['weeks']}",
                                affected_scenes=[],
                                actor_restriction={
                                    'actor': actor_name, 
                                    'available_weeks': actor_info['weeks']
                                }
                            ))
                        
                        # Parse daily restrictions
                        if actor_info.get('days') is not None:
                            constraints.append(Constraint(
                                source=ConstraintPriority.ACTOR,
                                type=constraint_type,
                                description=f"{actor_name} needs {actor_info['days']} days",
                                affected_scenes=[],
                                actor_restriction={
                                    'actor': actor_name, 
                                    'required_days': actor_info['days']
                                }
                            ))
            
            # NEW: Store cast_mapping for use by GA
            if 'cast_mapping' in people_data:
                cast_mapping_data = people_data['cast_mapping']
                print(f"DEBUG: Found cast_mapping with {len(cast_mapping_data)} character mappings")
                
                # Store cast_mapping in a special constraint for retrieval
                constraints.append(Constraint(
                    source=ConstraintPriority.ACTOR,
                    type=ConstraintType.SOFT,  # Not really a constraint, just data
                    description="Cast mapping data",
                    affected_scenes=[],
                    actor_restriction={
                        'cast_mapping': cast_mapping_data
                    }
                ))
        
        except Exception as e:
            print(f"DEBUG: Error parsing people constraints: {e}")
        
        return constraints
    
    def _parse_location_constraints(self, location_data: Dict) -> List[Constraint]:
        """Parse location availability and travel time constraints"""
        constraints = []
        
        try:
            # Parse location availability
            if 'locations' in location_data:
                locations_info = location_data['locations']
                
                if isinstance(locations_info, dict):
                    for location_name, location_info in locations_info.items():
                        if isinstance(location_info, dict) and 'constraints' in location_info:
                            for constraint_info in location_info['constraints']:
                                if isinstance(constraint_info, dict):
                                    constraint_level = constraint_info.get('constraint_level', 'Hard')
                                    constraint_type = ConstraintType.HARD if constraint_level == 'Hard' else ConstraintType.SOFT
                                    
                                    constraints.append(Constraint(
                                        source=ConstraintPriority.LOCATION,
                                        type=constraint_type,
                                        description=f"{location_name}: {constraint_info.get('details', '')}",
                                        affected_scenes=[],
                                        location_restriction={
                                            'location': location_name,
                                            'category': constraint_info.get('category'),
                                            'details': constraint_info.get('details')
                                        }
                                    ))
            
            # Parse travel times
            if 'travel_times' in location_data:
                travel_data = location_data['travel_times']
                
                if isinstance(travel_data, list):
                    for travel_info in travel_data:
                        if isinstance(travel_info, dict):
                            constraints.append(Constraint(
                                source=ConstraintPriority.LOCATION,
                                type=ConstraintType.SOFT,
                                description=f"Travel time: {travel_info.get('estimated_travel_time_minutes', 0)} minutes",
                                affected_scenes=[],
                                location_restriction={
                                    'from_location': travel_info.get('from_location_fictional', ''),
                                    'to_location': travel_info.get('to_location_fictional', ''),
                                    'travel_time_minutes': travel_info.get('estimated_travel_time_minutes', 0)
                                }
                            ))
        
        except Exception as e:
            print(f"DEBUG: Error parsing location constraints: {e}")
        
        return constraints
    
    def _parse_technical_constraints(self, technical_data: Dict) -> List[Constraint]:
        """Parse equipment and special requirements"""
        constraints = []
        
        try:
            # Parse equipment constraints
            if 'equipment' in technical_data:
                equipment_data = technical_data['equipment']
                
                for equipment_name, equipment_info in equipment_data.items():
                    constraint_level = equipment_info.get('constraint_level', 'Hard')
                    constraint_type = ConstraintType.HARD if constraint_level == 'Hard' else ConstraintType.SOFT
                    
                    constraints.append(Constraint(
                        source=ConstraintPriority.EQUIPMENT,
                        type=constraint_type,
                        description=f"{equipment_name}: {equipment_info.get('notes', '')}",
                        affected_scenes=[],
                        location_restriction={
                            'equipment': equipment_name,
                            'rental_type': equipment_info.get('type'),
                            'available_weeks': equipment_info.get('weeks', []),
                            'available_dates': equipment_info.get('dates', []),
                            'required_days': equipment_info.get('days')
                        }
                    ))
            
            # Parse special scene requirements
            if 'special_requirements' in technical_data:
                special_data = technical_data['special_requirements']
                
                for req_name, req_info in special_data.items():
                    constraint_level = req_info.get('constraint_level', 'Hard')
                    constraint_type = ConstraintType.HARD if constraint_level == 'Hard' else ConstraintType.SOFT
                    
                    # Extract scene numbers from notes
                    notes = req_info.get('notes', '')
                    scene_numbers = self._extract_scene_numbers(notes)
                    
                    constraints.append(Constraint(
                        source=ConstraintPriority.PREP_WRAP,
                        type=constraint_type,
                        description=f"{req_name}: {req_info.get('type')} - {notes}",
                        affected_scenes=scene_numbers,
                        date_restriction={
                            'prep_days': req_info.get('days'),
                            'department': req_info.get('type')
                        }
                    ))
        
        except Exception as e:
            print(f"DEBUG: Error parsing technical constraints: {e}")
        
        return constraints
    
    def _parse_creative_constraints(self, creative_data: Dict) -> List[Constraint]:
        """Parse director and DOP constraints"""
        constraints = []
        
        try:
            # Parse director constraints
            if 'director_notes' in creative_data:
                director_data = creative_data['director_notes']
                
                if isinstance(director_data, dict) and 'director_constraints' in director_data:
                    director_constraints = director_data['director_constraints']
                elif isinstance(director_data, list):
                    director_constraints = director_data
                else:
                    director_constraints = []
                
                if isinstance(director_constraints, list):
                    for constraint_info in director_constraints:
                        if isinstance(constraint_info, dict):
                            constraint_level = constraint_info.get('constraint_level', 'Hard')
                            constraint_type = ConstraintType.HARD if constraint_level == 'Hard' else ConstraintType.SOFT
                            
                            constraints.append(Constraint(
                                source=ConstraintPriority.DIRECTOR,
                                type=constraint_type,
                                description=constraint_info.get('constraint_text', ''),
                                affected_scenes=[str(s) for s in constraint_info.get('related_scenes', [])],
                                date_restriction={
                                    'category': constraint_info.get('category'),
                                    'locations': constraint_info.get('related_locations', [])
                                }
                            ))
            
            # Parse DOP constraints
            if 'dop_priorities' in creative_data:
                dop_data = creative_data['dop_priorities']
                
                if isinstance(dop_data, dict) and 'dop_priorities' in dop_data:
                    dop_constraints = dop_data['dop_priorities']
                elif isinstance(dop_data, list):
                    dop_constraints = dop_data
                else:
                    dop_constraints = []
                
                if isinstance(dop_constraints, list):
                    for constraint_info in dop_constraints:
                        if isinstance(constraint_info, dict):
                            constraint_level = constraint_info.get('constraint_level', 'Hard')
                            constraint_type = ConstraintType.HARD if constraint_level == 'Hard' else ConstraintType.SOFT
                            
                            constraints.append(Constraint(
                                source=ConstraintPriority.DOP,
                                type=constraint_type,
                                description=constraint_info.get('constraint_text', ''),
                                affected_scenes=[str(s) for s in constraint_info.get('related_scenes', [])],
                                location_restriction={
                                    'category': constraint_info.get('category'),
                                    'locations': constraint_info.get('related_locations', [])
                                }
                            ))
        
        except Exception as e:
            print(f"DEBUG: Error parsing creative constraints: {e}")
        
        return constraints
    
    def _parse_operational_data(self, operational_data: Dict) -> List[Constraint]:
        """Parse production rules and weather data"""
        constraints = []
        
        try:
            # Parse production rules
            if 'production_rules' in operational_data:
                prod_data = operational_data['production_rules']
                
                if 'rules' in prod_data:
                    production_rules = prod_data['rules']
                else:
                    production_rules = prod_data
                
                for rule_info in production_rules:
                    constraint_level = rule_info.get('constraint_level', 'Hard')
                    constraint_type = ConstraintType.HARD if constraint_level == 'Hard' else ConstraintType.SOFT
                    
                    constraints.append(Constraint(
                        source=ConstraintPriority.PRODUCTION,
                        type=constraint_type,
                        description=f"Production rule: {rule_info.get('raw_text', '')}",
                        affected_scenes=[],
                        date_restriction={
                            'parameter': rule_info.get('parameter_name'),
                            'rule_text': rule_info.get('raw_text')
                        }
                    ))
            
            # Parse weather data
            if 'weather' in operational_data:
                weather_data = operational_data['weather']
                
                if 'weekly_forecasts' in weather_data:
                    for week_key, week_info in weather_data['weekly_forecasts'].items():
                        constraints.append(Constraint(
                            source=ConstraintPriority.WEATHER,
                            type=ConstraintType.SOFT,
                            description=f"{week_key}: {week_info.get('weather_outlook', '')}",
                            affected_scenes=[],
                            date_restriction={
                                'week': week_key,
                                'date_range': week_info.get('date_range'),
                                'weather_outlook': week_info.get('weather_outlook')
                            }
                        ))
        
        except Exception as e:
            print(f"DEBUG: Error parsing operational constraints: {e}")
        
        return constraints
    
    def _extract_scene_numbers(self, text: str) -> List[str]:
        """Extract scene numbers from text"""
        if not text:
            return []
        
        # Look for "Scenes: [1, 2, 3]" pattern
        scene_match = re.search(r'Scenes:\s*\[([^\]]+)\]', text)
        if scene_match:
            scene_str = scene_match.group(1)
            return [s.strip() for s in scene_str.split(',')]
        
        # Look for individual scene numbers
        scene_numbers = re.findall(r'[Ss]cene\s+(\d+[a-z]?)', text)
        return scene_numbers

class ShootingCalendar:
    """Manages shooting dates and availability"""
    
    def __init__(self, start_date: str, end_date: str):
        self.start = datetime.strptime(start_date, "%Y-%m-%d")
        self.end = datetime.strptime(end_date, "%Y-%m-%d")
        self.shooting_days = self._generate_shooting_days()
    
    def _generate_shooting_days(self) -> List[date]:
        """Generate list of working days (excluding Sundays)"""
        days = []
        current = self.start
        
        while current <= self.end:
            if current.strftime("%A") != "Sunday":
                days.append(current.date())
            current += timedelta(days=1)
        
        return days
    
    def get_day_index(self, date_str: str) -> Optional[int]:
        """Convert date string to shooting day index"""
        try:
            target_date = datetime.strptime(date_str, "%Y-%m-%d").date()
            if target_date in self.shooting_days:
                return self.shooting_days.index(target_date)
        except:
            pass
        return None

class LocationClusterManager:
    """ENHANCED: Manages grouping of scenes by geographic location with REAL time estimation"""
    
    def __init__(self, stripboard: List[Dict], scene_time_estimates: Dict[str, float]):
        self.stripboard = stripboard
        self.scene_time_estimates = scene_time_estimates or {}   # NEW: Real time estimates # Handle None case
        self.clusters = self._create_location_clusters()
        
        print(f"DEBUG: Created {len(self.clusters)} location clusters using REAL time estimates")
        for i, cluster in enumerate(self.clusters):
            print(f"  Cluster {i}: {cluster.location} - {len(cluster.scenes)} scenes, {cluster.estimated_days} days, {cluster.total_hours:.1f} hours")
    
    def _create_location_clusters(self) -> List[LocationCluster]:
        """Group scenes by geographic location with REAL scene time estimation"""
        location_groups = defaultdict(list)
    
        for scene in self.stripboard:
            location = scene.get('Geographic_Location', 'Unknown Location')
            # Skip "Location TBD" or empty locations
            if location and location != 'Location TBD':
                location_groups[location].append(scene)
    
        clusters = []
        for location, scenes in location_groups.items():
            # FIXED: Use REAL scene time estimates instead of page count math
            total_hours = 0.0
            for scene in scenes:
                '''scene_number = str(scene.get('Scene_Number', ''))
                
                # Get real time estimate or use fallback
                if scene_number in self.scene_time_estimates:
                    scene_hours = self.scene_time_estimates[scene_number]
                    print(f"DEBUG: Scene {scene_number} - {scene_hours} hours (from estimates)")
                else:
                    # Fallback to page count estimation only if no real estimate
                    scene_hours = self._estimate_scene_hours_from_page_count(scene)
                    print(f"DEBUG: Scene {scene_number} - {scene_hours} hours (fallback from page count)")
                '''
                scene_number = scene.get('Scene_Number', '')

                # Get real time estimate or use fallback - try multiple formats
                if str(scene_number) in self.scene_time_estimates:
                    scene_hours = self.scene_time_estimates[str(scene_number)]
                    print(f"DEBUG: Scene {scene_number} - {scene_hours} hours (from estimates)")
                elif scene_number in self.scene_time_estimates:
                    scene_hours = self.scene_time_estimates[scene_number]
                    print(f"DEBUG: Scene {scene_number} - {scene_hours} hours (from estimates)")
                else:
                    # Fallback to page count estimation only if no real estimate
                    scene_hours = self._estimate_scene_hours_from_page_count(scene)
                    print(f"DEBUG: Scene {scene_number} - {scene_hours} hours (fallback from page count)")

                total_hours += scene_hours
        
            # FIXED: Convert to shooting days using DAILY HOUR LIMITS (10 hours per day)
            MAX_HOURS_PER_DAY = 10.0
            estimated_days = max(1, int((total_hours + MAX_HOURS_PER_DAY - 0.1) / MAX_HOURS_PER_DAY))  # Ceiling division
            
            # CONSTRAINT: Still limit clusters to reasonable sizes (max 4 days per location)
            estimated_days = min(estimated_days, 4)
        
            # Extract unique actors
            all_actors = set()
            for scene in scenes:
                cast = scene.get('Cast', [])
                if isinstance(cast, list):
                    all_actors.update(cast)
                elif isinstance(cast, str) and cast:
                    all_actors.add(cast)
        
            clusters.append(LocationCluster(
                location=location,
                scenes=scenes,
                total_hours=total_hours,  # FIXED: Use proper field name
                estimated_days=estimated_days,
                required_actors=list(all_actors)
            ))
    
        # Sort clusters by total hours (larger first for better scheduling)
        clusters.sort(key=lambda x: x.total_hours, reverse=True)
        return clusters
    
    def _estimate_scene_hours_from_page_count(self, scene: Dict) -> float:
        """Fallback: Estimate scene hours from page count when real estimates unavailable"""
        page_count = scene.get('Page_Count', '1')
        
        # Base time estimation from page count
        if isinstance(page_count, str):
            time_multiplier = self._parse_page_count(page_count)
        else:
            time_multiplier = 1.0
        
        base_hours = 1.0 * time_multiplier  # 1 hour per page as base
        
        # Adjust based on scene characteristics (keep existing logic as fallback)
        cast = scene.get('Cast', [])
        cast_size = len(cast) if isinstance(cast, list) else 1
        if cast_size > 3:
            base_hours *= 1.5  # More cast = more time
        
        # Adjust based on INT/EXT
        if scene.get('INT_EXT') == 'EXT':
            base_hours *= 1.3  # Exterior scenes take longer
        
        return base_hours
    
    def _parse_page_count(self, page_count_str: str) -> float:
        """Parse page count strings like '1 6/8', '3/8', '2 1/8' into decimal multipliers"""
        try:
            # Handle formats like "1 6/8", "3/8", "2"
            if '/' in page_count_str:
                parts = page_count_str.strip().split()
                if len(parts) == 2:  # "1 6/8"
                    whole = int(parts[0])
                    frac_parts = parts[1].split('/')
                    fraction = int(frac_parts[0]) / int(frac_parts[1])
                    return whole + fraction
                else:  # "6/8"
                    frac_parts = page_count_str.split('/')
                    return int(frac_parts[0]) / int(frac_parts[1])
            else:  # "2"
                return float(page_count_str.strip())
        except:
            return 1.0  # Default if parsing fails

class LocationFirstGA:
    """Location-First Genetic Algorithm for scheduling"""
    
    def __init__(self, cluster_manager: LocationClusterManager, constraints: List[Constraint], 
             calendar: ShootingCalendar, params: Dict, cast_mapping: Dict[str, str]):
        """Initialize GA - WITH CAST MAPPING"""
        self.cluster_manager = cluster_manager
        self.constraints = constraints
        self.calendar = calendar
        self.params = params
        self.rng = np.random.RandomState(params.get('seed', 42))
        
        # NEW: Store cast mapping from n8n
        self.cast_mapping = cast_mapping
        print(f"DEBUG: GA received cast_mapping: {self.cast_mapping}")
    
    # Build constraint maps for efficient lookup
    self._build_constraint_maps()
    self._build_travel_times()
    
    # STEP 2.5a: COMPLETE ACTOR CONSTRAINTS
    # Replace these methods in LocationFirstGA class

    def _build_constraint_maps(self):
        """Build efficient lookup structures for constraints - USING N8N CAST MAPPING"""
        # Actor constraint data structures (unchanged)
        self.actor_unavailable_dates = defaultdict(list)
        self.actor_available_weeks = {}
        self.actor_required_days = {}
        self.actor_constraint_types = {}
        self.actor_constraint_levels = {}
        
        # REMOVED: No more hard-coded mapping
        # self.actor_to_character_map = {}  # DELETED
        # self._build_actor_character_mapping()  # DELETED
        
        # Other constraint maps (unchanged)
        self.director_mandates = {}
        self.location_windows = {}
        
        for constraint in self.constraints:
            if constraint.actor_restriction:
                self._map_actor_constraint(constraint)
            elif constraint.source == ConstraintPriority.DIRECTOR and constraint.affected_scenes:
                for scene in constraint.affected_scenes:
                    self.director_mandates[scene] = constraint
            elif constraint.location_restriction and constraint.location_restriction.get('location'):
                location = constraint.location_restriction['location']
                self.location_windows[location] = constraint
        
        self._build_travel_times()
    
    def _build_travel_times(self):
        """Build travel time matrix between locations"""
        locations = [cluster.location for cluster in self.cluster_manager.clusters]
        n = len(locations)
        self.travel_matrix = np.zeros((n, n))
        
        # Find travel time constraints
        travel_constraints = [c for c in self.constraints 
                             if c.location_restriction and 'travel_time_minutes' in c.location_restriction]
        
        for i, loc1 in enumerate(locations):
            for j, loc2 in enumerate(locations):
                if i != j:
                    # Look for specific travel time, otherwise estimate
                    travel_time = self._find_travel_time(loc1, loc2, travel_constraints)
                    if travel_time is None:
                        travel_time = self._estimate_travel_time(loc1, loc2)
                    
                    # Convert minutes to penalty points (assuming 1 mile = 2 minutes average)
                    miles = travel_time / 2.0
                    self.travel_matrix[i][j] = miles
    
    def _find_travel_time(self, loc1: str, loc2: str, travel_constraints: List) -> Optional[float]:
        """Find specific travel time between locations"""
        for constraint in travel_constraints:
            restriction = constraint.location_restriction
        
            # Get values with None check
            from_loc = restriction.get('from_location_fictional')
            to_loc = restriction.get('to_location_fictional')
        
            # Only check if both values exist and are strings
            if from_loc and to_loc and isinstance(from_loc, str) and isinstance(to_loc, str):
                if ((from_loc in loc1 or to_loc in loc1) and 
                    (from_loc in loc2 or to_loc in loc2)):
                    return restriction.get('travel_time_minutes', 30)
    
        return None
    
    def _estimate_travel_time(self, loc1: str, loc2: str) -> float:
        """Estimate travel time between locations based on addresses"""
        if loc1 == loc2:
            return 0.0
        
        # Simple city/state comparison
        def extract_city_state(address):
            city_match = re.search(r',\s*([^,]+),\s*([A-Z]{2})', address)
            return city_match.groups() if city_match else (None, None)
        
        city1, state1 = extract_city_state(loc1)
        city2, state2 = extract_city_state(loc2)
        
        if city1 and city2:
            if city1 == city2:
                return 15  # Same city - 15 minutes
            elif state1 == state2:
                return 60  # Same state - 1 hour
            else:
                return 180  # Different states - 3 hours
        
        return 60  # Default estimate
    
    def create_individual(self) -> Dict:
        """
        Create individual with CONSECUTIVE scheduling starting from Day 1
        Individual = {
            'sequence': [cluster_indices],  # Order to visit location clusters
            'day_assignments': [day_index_per_cluster]  # Which day each cluster starts
        }
        """
        n_clusters = len(self.cluster_manager.clusters)
        n_days = len(self.calendar.shooting_days)
        
        # Random sequence of location clusters
        sequence = list(range(n_clusters))
        self.rng.shuffle(sequence)
        
        # FIXED: Assign clusters to CONSECUTIVE days starting from Day 1 (index 0)
        day_assignments = []
        current_day = 0  # Start from Day 1 (index 0)
        
        for cluster_idx in sequence:
            cluster = self.cluster_manager.clusters[cluster_idx]
            
            # Assign current day to this cluster
            day_assignments.append(current_day)
            
            # Move to next available day after this cluster completes
            cluster_duration = max(1, cluster.estimated_days)
            current_day += cluster_duration
            
            # If we exceed calendar, compress remaining clusters
            if current_day >= n_days:
                # Compress remaining clusters into available days
                remaining_clusters = len(sequence) - len(day_assignments)
                if remaining_clusters > 0:
                    days_per_remaining = max(1, (n_days - current_day + cluster_duration) // remaining_clusters)
                    current_day = max(0, n_days - (remaining_clusters * days_per_remaining))
        
        return {
            'sequence': sequence,
            'day_assignments': day_assignments
        }
    
    def fitness(self, individual: Dict) -> float:
        """Calculate fitness using graduated penalty system - WITH STEP 2.5a DEBUG"""
        score = 0.0
        
        sequence = individual['sequence']
        day_assignments = individual['day_assignments']
        
        # 1. ENHANCED: Actor constraint violations (Step 2.5a)
        actor_violations = self._check_complete_actor_violations(sequence, day_assignments)
        actor_penalty = PENALTY_HARD_CONSTRAINT * actor_violations
        score += actor_penalty
        
        # 2. EXISTING: Other penalties (unchanged for now)
        location_splits = self._count_location_splits(sequence, day_assignments)
        score += PENALTY_LOCATION_SPLIT * location_splits
        
        director_violations = self._count_director_violations(sequence, day_assignments)
        score += PENALTY_DIRECTOR_MANDATE * director_violations
        
        travel_penalty = self._calculate_travel_penalty(sequence)
        score += travel_penalty
        
        idle_penalty = self._calculate_actor_idle_penalty(sequence, day_assignments)
        score += idle_penalty
        
        soft_bonus = self._calculate_soft_bonus(sequence, day_assignments)
        score += soft_bonus
        
        # DEBUG: Print constraint breakdown for Step 2.5a
        if actor_violations > 0:
            print(f"DEBUG: Fitness breakdown - Actor violations: {actor_violations}, penalty: {actor_penalty}")
        
        return score

    
    def _count_hard_violations(self, sequence: List[int], day_assignments: List[int]) -> int:
        """Calculate hard constraint violations - EXTENDED for complete actor checking"""
        violations = 0
        
        # EXTENDED: Complete actor constraint checking
        actor_violations = self._check_complete_actor_violations(sequence, day_assignments)
        violations += actor_violations
        
        # EXISTING: Other violations (keep as-is for now)
        location_splits = self._count_location_splits(sequence, day_assignments)
        violations += location_splits  # This returns 0 anyway
        
        director_violations = self._count_director_violations(sequence, day_assignments)
        violations += director_violations  # This returns 0 anyway
        
        travel_penalty = self._calculate_travel_penalty(sequence)
        # Note: travel_penalty is negative, not a violation count
        
        idle_penalty = self._calculate_actor_idle_penalty(sequence, day_assignments)
        # Note: idle_penalty is negative, not a violation count
        
        if actor_violations > 0:
            print(f"DEBUG: Found {actor_violations} actor constraint violations")
        
    def _map_actor_constraint(self, constraint):
        """NEW: Map complete actor constraint data from n8n"""
        actor_restriction = constraint.actor_restriction
        actor = actor_restriction.get('actor')
        
        if not actor:
            return
        
        # Store constraint metadata
        self.actor_constraint_levels[actor] = constraint.type.value
        
        # EXISTING: Unavailable dates (keep working code)
        unavailable_date = actor_restriction.get('unavailable_date')
        if unavailable_date:
            self.actor_unavailable_dates[actor].append(unavailable_date)
        
        # NEW: Available weeks
        available_weeks = actor_restriction.get('available_weeks')
        if available_weeks:
            self.actor_available_weeks[actor] = available_weeks
            print(f"DEBUG: Actor {actor} only available weeks: {available_weeks}")
        
        # NEW: Required days
        required_days = actor_restriction.get('required_days')
        if required_days:
            self.actor_required_days[actor] = required_days
            print(f"DEBUG: Actor {actor} needs {required_days} days") 

    def _check_complete_actor_violations(self, sequence: List[int], day_assignments: List[int]) -> int:
        """EXTENDED: Check ALL actor constraint types from n8n"""
        violations = 0
        
        # EXISTING: Check unavailable dates (keep working code)
        violations += self._check_actor_unavailable_dates(sequence, day_assignments)
        
        # NEW: Check available weeks
        violations += self._check_actor_available_weeks(sequence, day_assignments)
        
        # NEW: Check required days
        violations += self._check_actor_required_days(sequence, day_assignments)
        
        return violations           

    def _check_actor_unavailable_dates(self, sequence: List[int], day_assignments: List[int]) -> int:
        """NEW: Check unavailable dates using n8n cast_mapping"""
        violations = 0
        
        for i, cluster_idx in enumerate(sequence):
            cluster = self.cluster_manager.clusters[cluster_idx]
            start_day = day_assignments[i]
            
            # Check each day this cluster needs
            for day_offset in range(cluster.estimated_days):
                shooting_day_idx = start_day + day_offset
                if shooting_day_idx >= len(self.calendar.shooting_days):
                    violations += 1
                    continue
                
                shooting_date = self.calendar.shooting_days[shooting_day_idx]
                
                # Check each character in this cluster
                for character in cluster.required_actors:
                    actor = self._get_actor_for_character(character)
                    
                    if actor and actor in self.actor_unavailable_dates:
                        for unavailable_date_str in self.actor_unavailable_dates[actor]:
                            try:
                                unavailable_date = datetime.strptime(unavailable_date_str, "%Y-%m-%d").date()
                                if shooting_date == unavailable_date:
                                    violations += 1
                                    print(f"DEBUG: VIOLATION - Actor '{actor}' (character '{character}') unavailable on {shooting_date}")
                            except:
                                pass
        
        return violations    

    def _check_actor_available_weeks(self, sequence: List[int], day_assignments: List[int]) -> int:
        """FIXED: Check available weeks using n8n cast_mapping"""
        violations = 0
        
        for i, cluster_idx in enumerate(sequence):
            cluster = self.cluster_manager.clusters[cluster_idx]
            start_day = day_assignments[i]
            
            # Check each character in this cluster
            for character in cluster.required_actors:
                actor = self._get_actor_for_character(character)
                
                if actor and actor in self.actor_available_weeks:
                    available_weeks = self.actor_available_weeks[actor]
                    
                    # Check if any day of this cluster falls outside available weeks
                    for day_offset in range(cluster.estimated_days):
                        day_idx = start_day + day_offset
                        if day_idx < len(self.calendar.shooting_days):
                            day_week = self._get_shooting_week_from_day(day_idx)
                            
                            if day_week not in available_weeks:
                                violations += 1
                                print(f"DEBUG: VIOLATION - Actor '{actor}' (character '{character}') scheduled in week {day_week}, only available weeks {available_weeks}")
                                break
        
        return violations

    def _check_actor_required_days(self, sequence: List[int], day_assignments: List[int]) -> int:
        """FIXED: Check actor days using n8n cast_mapping"""
        violations = 0
        
        # Count actual shooting days per character
        character_scheduled_days = defaultdict(int)
        
        for i, cluster_idx in enumerate(sequence):
            cluster = self.cluster_manager.clusters[cluster_idx]
            start_day = day_assignments[i]
            
            # Count days for each character in this cluster
            for character in cluster.required_actors:
                for day_offset in range(cluster.estimated_days):
                    day_idx = start_day + day_offset
                    if day_idx < len(self.calendar.shooting_days):
                        character_scheduled_days[character] += 1
        
        print(f"DEBUG: Characters with scheduled days: {dict(character_scheduled_days)}")
        
        # Check against required days (actor constraints)
        for actor, required_days in self.actor_required_days.items():
            # Find character(s) this actor plays
            characters_for_actor = [char for char, mapped_actor in self.cast_mapping.items() 
                                if mapped_actor == actor]
            
            total_scheduled_days = 0
            for character in characters_for_actor:
                total_scheduled_days += character_scheduled_days.get(character, 0)
            
            if total_scheduled_days != required_days:
                violations += 1
                print(f"DEBUG: VIOLATION - Actor '{actor}' (characters {characters_for_actor}) scheduled {total_scheduled_days} days, needs {required_days}")
            else:
                print(f"DEBUG: OK - Actor '{actor}' (characters {characters_for_actor}) scheduled {total_scheduled_days} days, needs {required_days}")
        
        return violations

    def _get_actor_for_character(self, character_name: str) -> Optional[str]:
        """NEW: Get actor name from character name using n8n cast_mapping"""
        actor_name = self.cast_mapping.get(character_name)
        if actor_name:
            print(f"DEBUG: Character '{character_name}' maps to actor '{actor_name}'")
            return actor_name
        
        print(f"DEBUG: No actor mapping found for character '{character_name}'")
        return None

    def _count_location_splits(self, sequence: List[int], day_assignments: List[int]) -> int:
        """Count how many locations are split across multiple non-consecutive days"""
        # In this implementation, each cluster is assigned to consecutive days
        # so we don't have location splits. This is more of a penalty for
        # future algorithms that might split clusters.
        return 0
    
    def _count_director_violations(self, sequence: List[int], day_assignments: List[int]) -> int:
        """Count violations of director mandates"""
        violations = 0
        
        # Check for any scene-specific director requirements
        for constraint in self.constraints:
            if constraint.source == ConstraintPriority.DIRECTOR and constraint.affected_scenes:
                # This is a simplified check - in reality would need more complex logic
                # to check if director requirements are violated
                pass
        
        return violations
    
    def _calculate_travel_penalty(self, sequence: List[int]) -> float:
        """Calculate travel time penalty between consecutive location clusters"""
        penalty = 0.0
        
        for i in range(len(sequence) - 1):
            from_cluster_idx = sequence[i]
            to_cluster_idx = sequence[i + 1]
            
            travel_miles = self.travel_matrix[from_cluster_idx][to_cluster_idx]
            penalty += PENALTY_TRAVEL_PER_MILE * travel_miles
        
        return penalty
    
    def _calculate_actor_idle_penalty(self, sequence: List[int], day_assignments: List[int]) -> float:
        """Calculate penalty for actor idle days"""
        penalty = 0.0
        
        # Build actor working days
        actor_working_days = defaultdict(set)
        
        for i, cluster_idx in enumerate(sequence):
            cluster = self.cluster_manager.clusters[cluster_idx]
            start_day = day_assignments[i]
            
            for actor in cluster.required_actors:
                for day_offset in range(cluster.estimated_days):
                    shooting_day = start_day + day_offset
                    if shooting_day < len(self.calendar.shooting_days):
                        actor_working_days[actor].add(shooting_day)
        
        # Calculate idle days for each actor
        for actor, working_days in actor_working_days.items():
            if len(working_days) > 1:
                working_days_list = sorted(list(working_days))
                first_day = working_days_list[0]
                last_day = working_days_list[-1]
                total_span = last_day - first_day + 1
                actual_working_days = len(working_days)
                idle_days = total_span - actual_working_days
                
                penalty += PENALTY_ACTOR_IDLE_DAY * idle_days
        
        return penalty
    
    def _calculate_soft_bonus(self, sequence: List[int], day_assignments: List[int]) -> float:
        """Calculate bonus for satisfied soft constraints"""
        bonus = 0.0
        
        # Count satisfied soft constraints
        soft_constraints = [c for c in self.constraints if c.type == ConstraintType.SOFT]
        
        for constraint in soft_constraints:
            # Simplified check - assume soft constraints are mostly satisfied
            # unless there are obvious violations
            if self._is_soft_constraint_satisfied(constraint, sequence, day_assignments):
                bonus += BONUS_SOFT_CONSTRAINT
        
        return bonus
    
    def _is_soft_constraint_satisfied(self, constraint: Constraint, sequence: List[int], 
                                     day_assignments: List[int]) -> bool:
        """Check if a soft constraint is satisfied"""
        # Simplified implementation - assume most soft constraints are satisfied
        # unless there are clear violations
        return True
    
    def evolve(self) -> Tuple[Dict, float]:
        """Run genetic algorithm for location-first scheduling"""
        pop_size = self.params.get('phase1_population', 50)
        generations = self.params.get('phase1_generations', 200)
        
        # Initialize population
        population = [self.create_individual() for _ in range(pop_size)]
        best_individual = None
        best_fitness = -float('inf')
        
        print(f"DEBUG: Starting evolution with {pop_size} individuals for {generations} generations")
        
        for gen in range(generations):
            # Evaluate fitness
            fitnesses = []
            for ind in population:
                fitness = self.fitness(ind)
                fitnesses.append(fitness)
            
            # Track best
            gen_best_idx = np.argmax(fitnesses)
            if fitnesses[gen_best_idx] > best_fitness:
                best_fitness = fitnesses[gen_best_idx]
                best_individual = population[gen_best_idx].copy()
            
            if gen % 50 == 0:
                print(f"DEBUG: Generation {gen}, Best fitness: {best_fitness}")
            
            # Create new population
            new_population = []
            
            # Elitism - keep best individuals
            elite_count = max(2, pop_size // 10)
            elite_indices = np.argsort(fitnesses)[-elite_count:]
            for idx in elite_indices:
                new_population.append(self._copy_individual(population[idx]))
            
            # Generate rest through selection and reproduction
            while len(new_population) < pop_size:
                # Tournament selection
                parent1 = self._tournament_selection(population, fitnesses)
                parent2 = self._tournament_selection(population, fitnesses)
                
                # Crossover
                child = self._crossover(parent1, parent2)
                
                # Mutation
                child = self._mutate(child)
                
                new_population.append(child)
            
            population = new_population[:pop_size]
        
        print(f"DEBUG: Evolution completed. Best fitness: {best_fitness}")
        return best_individual, best_fitness
    
    def _copy_individual(self, individual: Dict) -> Dict:
        """Create a copy of an individual"""
        return {
            'sequence': individual['sequence'].copy(),
            'day_assignments': individual['day_assignments'].copy()
        }
    
    def _tournament_selection(self, population: List, fitnesses: List) -> Dict:
        """Tournament selection"""
        tournament_size = 5
        indices = self.rng.choice(len(population), tournament_size, replace=False)
        tournament_fits = [fitnesses[i] for i in indices]
        winner_idx = indices[np.argmax(tournament_fits)]
        return self._copy_individual(population[winner_idx])
    
    def _crossover(self, parent1: Dict, parent2: Dict) -> Dict:
        """Crossover for location sequence and day assignments"""
        if self.rng.random() > self.params.get('crossover_rate', 0.85):
            return self._copy_individual(parent1)
        
        n_clusters = len(self.cluster_manager.clusters)
        
        # Order crossover for sequence
        child_sequence = self._order_crossover(parent1['sequence'], parent2['sequence'])
        
        # Uniform crossover for day assignments
        child_day_assignments = []
        for i in range(len(parent1['day_assignments'])):
            if self.rng.random() < 0.5:
                child_day_assignments.append(parent1['day_assignments'][i])
            else:
                child_day_assignments.append(parent2['day_assignments'][i])
        
        return {
            'sequence': child_sequence,
            'day_assignments': child_day_assignments
        }
    
    def _order_crossover(self, parent1: List[int], parent2: List[int]) -> List[int]:
        """Order crossover for sequence"""
        n = len(parent1)
        start, end = sorted(self.rng.choice(n, 2, replace=False))
        
        child = [-1] * n
        child[start:end] = parent1[start:end]
        
        pointer = end
        for gene in parent2[end:] + parent2[:end]:
            if gene not in child:
                child[pointer % n] = gene
                pointer += 1
        
        return child
    
    def _mutate(self, individual: Dict) -> Dict:
        """Mutation for individual"""
        mutation_rate = self.params.get('mutation_rate', 0.02)
        
        # Mutate sequence (swap two locations)
        if self.rng.random() < mutation_rate:
            sequence = individual['sequence']
            if len(sequence) > 1:
                i, j = self.rng.choice(len(sequence), 2, replace=False)
                sequence[i], sequence[j] = sequence[j], sequence[i]
        
        # Mutate day assignments (shift a cluster to different day)
        if self.rng.random() < mutation_rate:
            day_assignments = individual['day_assignments']
            if day_assignments:
                cluster_idx = self.rng.randint(len(day_assignments))
                max_day = len(self.calendar.shooting_days) - 1
                day_assignments[cluster_idx] = self.rng.randint(0, max_day + 1)
        
        return individual

class ScheduleOptimizer:
    """Main orchestrator for location-first optimization"""
    
    def __init__(self, request: ScheduleRequest):
        """Initialize optimizer - WITH CAST MAPPING EXTRACTION"""
        self.stripboard = request.stripboard
        self.constraints_raw = request.constraints
        self.params = request.ga_params
        
        # Initialize components
        self.parser = StructuredConstraintParser()
        self.constraints = self.parser.parse_all_constraints(self.constraints_raw)
        
        # NEW: Extract cast_mapping from constraints
        self.cast_mapping = self._extract_cast_mapping()
        
        # Create location cluster manager
        scene_time_estimates = self._get_scene_time_estimates()
        self.cluster_manager = LocationClusterManager(self.stripboard, scene_time_estimates)
        
        # Determine shooting calendar
        self.calendar = ShootingCalendar("2025-09-01", "2025-10-31")
        
        print(f"DEBUG: PARSED {len(self.constraints)} CONSTRAINTS")
        print(f"DEBUG: EXTRACTED {len(self.cast_mapping)} CHARACTER-ACTOR MAPPINGS")
        print(f"DEBUG: CREATED {len(self.cluster_manager.clusters)} LOCATION CLUSTERS")
        print(f"DEBUG: CALENDAR HAS {len(self.calendar.shooting_days)} SHOOTING DAYS")

    def _extract_cast_mapping(self) -> Dict[str, str]:
        """NEW: Extract cast_mapping from parsed constraints"""
        for constraint in self.constraints:
            if (constraint.actor_restriction and 
                'cast_mapping' in constraint.actor_restriction):
                
                cast_mapping_data = constraint.actor_restriction['cast_mapping']
                
                # Convert n8n format to simple dict
                cast_mapping = {}
                for character_name, character_info in cast_mapping_data.items():
                    actor_name = character_info.get('actor_name')
                    if actor_name:
                        cast_mapping[character_name] = actor_name
                        print(f"DEBUG: Character '{character_name}' → Actor '{actor_name}'")
                
                return cast_mapping
        
        print(f"DEBUG: No cast_mapping found in constraints")
        return {}
    
    def optimize(self) -> Dict[str, Any]:
        """Run location-first optimization - WITH CAST MAPPING"""
        import time
        start_time = time.time()
        
        # NEW: Pass cast_mapping to GA
        ga = LocationFirstGA(self.cluster_manager, self.constraints, self.calendar, 
                            self.params, self.cast_mapping)
        best_individual, best_fitness = ga.evolve()
        
        # Rest unchanged
        final_schedule = self._build_final_schedule(best_individual)
        schedule_summary = self._build_schedule_summary(final_schedule)
        metrics = self._calculate_metrics(final_schedule)
        missing_scenes_summary = {
            'total_missing_scenes': len(getattr(self, 'missing_scenes_summary', [])),
            'missing_scenes_details': getattr(self, 'missing_scenes_summary', [])
        }
        processing_time = time.time() - start_time
        
        return {
            'schedule': final_schedule,
            'summary': schedule_summary,
            'missing_scenes': missing_scenes_summary,
            'conflicts': [],
            'metrics': metrics,
            'fitness_score': best_fitness,
            'processing_time_seconds': processing_time
        }
    
    def _build_final_schedule(self, individual: Dict) -> List[Dict]:
        """Build final day-by-day schedule from best individual - WITH SCENE SPLITTING"""
        schedule = []
        
        sequence = individual['sequence']
        day_assignments = individual['day_assignments']
        
        # Track which scenes have been scheduled to prevent duplicates
        scheduled_scenes = set()
        
        # Track missing scenes with reasons
        missing_scenes = []
        
        # Force consecutive scheduling starting from Day 1 (index 0)
        current_day_idx = 0
        
        for i, cluster_idx in enumerate(sequence):
            cluster = self.cluster_manager.clusters[cluster_idx]
            
            # Always use consecutive days
            actual_start_day = current_day_idx
            
            # Get scenes that haven't been scheduled yet
            cluster_scenes = [scene for scene in cluster.scenes 
                            if scene['Scene_Number'] not in scheduled_scenes]
            
            if not cluster_scenes:
                continue  # Skip if all scenes already scheduled
            
            # FIXED: Distribute scenes by TIME with SCENE SPLITTING
            shooting_day_idx = actual_start_day
            current_day_scenes = []
            current_day_hours = 0.0
            MAX_DAILY_HOURS = 10.0
            
            for scene in cluster_scenes:
                # Check if we exceed calendar bounds
                if shooting_day_idx >= len(self.calendar.shooting_days):
                    # Track remaining scenes as missing
                    remaining_scenes = cluster_scenes[cluster_scenes.index(scene):]
                    for missing_scene in remaining_scenes:
                        if missing_scene['Scene_Number'] not in scheduled_scenes:
                            missing_scenes.append({
                                'scene_number': missing_scene['Scene_Number'],
                                'location_name': missing_scene.get('Location_Name', 'Unknown'),
                                'reason': 'Calendar overflow - exceeds available shooting days',
                                'geographic_location': missing_scene.get('Geographic_Location', 'Unknown')
                            })
                    break
                
                scene_hours = self._get_scene_hours(scene)
                
                # OPTION C: Handle scenes longer than daily limit with SPLITTING
                if scene_hours > MAX_DAILY_HOURS:
                    print(f"INFO: Splitting Scene {scene['Scene_Number']} ({scene_hours:.1f}h) across multiple days")
                    
                    # If we have scenes in current day, finish the day first
                    if current_day_scenes:
                        self._add_schedule_day(schedule, shooting_day_idx, current_day_scenes, cluster, scheduled_scenes)
                        shooting_day_idx += 1
                        current_day_scenes = []
                        current_day_hours = 0.0
                    
                    # Split the scene across multiple days
                    scene_parts = self._split_scene_into_parts(scene, scene_hours, MAX_DAILY_HOURS)
                    
                    for part_idx, scene_part in enumerate(scene_parts):
                        # Check calendar bounds for each part
                        if shooting_day_idx >= len(self.calendar.shooting_days):
                            # Track remaining parts as missing
                            remaining_parts = scene_parts[part_idx:]
                            for missing_part in remaining_parts:
                                missing_scenes.append({
                                    'scene_number': missing_part['Scene_Number'],
                                    'location_name': missing_part.get('Location_Name', 'Unknown'),
                                    'reason': 'Calendar overflow - scene parts exceed available shooting days',
                                    'geographic_location': missing_part.get('Geographic_Location', 'Unknown')
                                })
                            break
                        
                        # Schedule each part on its own day
                        self._add_schedule_day(schedule, shooting_day_idx, [scene_part], cluster, scheduled_scenes)
                        shooting_day_idx += 1
                    
                    continue  # Move to next scene
                
                # If adding this scene exceeds daily limit, start new day
                if current_day_hours + scene_hours > MAX_DAILY_HOURS and current_day_scenes:
                    # Finish current day
                    self._add_schedule_day(schedule, shooting_day_idx, current_day_scenes, cluster, scheduled_scenes)
                    shooting_day_idx += 1
                    
                    # Check calendar bounds before starting new day
                    if shooting_day_idx >= len(self.calendar.shooting_days):
                        # No more days available
                        remaining_scenes = cluster_scenes[cluster_scenes.index(scene):]
                        for missing_scene in remaining_scenes:
                            if missing_scene['Scene_Number'] not in scheduled_scenes:
                                missing_scenes.append({
                                    'scene_number': missing_scene['Scene_Number'],
                                    'location_name': missing_scene.get('Location_Name', 'Unknown'),
                                    'reason': 'Calendar overflow - exceeds available shooting days',
                                    'geographic_location': missing_scene.get('Geographic_Location', 'Unknown')
                                })
                        break
                    
                    # Start new day
                    current_day_scenes = [scene]
                    current_day_hours = scene_hours
                else:
                    # Add to current day
                    current_day_scenes.append(scene)
                    current_day_hours += scene_hours
            
            # Add final day if there are remaining scenes
            if current_day_scenes and shooting_day_idx < len(self.calendar.shooting_days):
                self._add_schedule_day(schedule, shooting_day_idx, current_day_scenes, cluster, scheduled_scenes)
                shooting_day_idx += 1
            
            # Update current_day_idx for next cluster (consecutive scheduling)
            current_day_idx = shooting_day_idx
        
        # Check for scenes that were never included in any cluster
        all_original_scenes = {scene['Scene_Number'] for scene in self.stripboard}
        all_clustered_scenes = set()
        for cluster in self.cluster_manager.clusters:
            for scene in cluster.scenes:
                all_clustered_scenes.add(scene['Scene_Number'])
        
        never_clustered_scenes = all_original_scenes - all_clustered_scenes
        for scene_num in never_clustered_scenes:
            scene_data = next((s for s in self.stripboard if s['Scene_Number'] == scene_num), None)
            if scene_data:
                geo_location = scene_data.get('Geographic_Location', 'Unknown')
                
                if geo_location in ['Location TBD', '', None, 'Unknown Location']:
                    reason = 'No valid geographic location - Location TBD or empty'
                else:
                    reason = 'Failed to cluster - unknown issue'
                    
                missing_scenes.append({
                    'scene_number': scene_num,
                    'location_name': scene_data.get('Location_Name', 'Unknown'),
                    'reason': reason,
                    'geographic_location': geo_location
                })
        
        # Store missing scenes for reporting
        self.missing_scenes_summary = missing_scenes
        
        return schedule

    def _get_scene_hours(self, scene: Dict) -> float:
        """Get estimated hours for a single scene - handles both original and split scenes"""
        
        # Check if this is a split scene part
        if 'split_info' in scene and scene['split_info'].get('is_split'):
            return scene['split_info']['part_hours']
        
        # Original scene - get from estimates
        scene_number = scene.get('Scene_Number', '')
        
        # Try to get from scene time estimates
        if str(scene_number) in self.cluster_manager.scene_time_estimates:
            return self.cluster_manager.scene_time_estimates[str(scene_number)]
        elif scene_number in self.cluster_manager.scene_time_estimates:
            return self.cluster_manager.scene_time_estimates[scene_number]
        else:
            # Fallback to page count estimation
            return self._estimate_scene_hours_from_page_count(scene)


    def _estimate_scene_hours_from_page_count(self, scene: Dict) -> float:
        """Fallback: Estimate scene hours from page count when real estimates unavailable"""
        page_count = scene.get('Page_Count', '1')
        
        # Parse page count
        if isinstance(page_count, str):
            time_multiplier = self._parse_page_count(page_count)
        else:
            time_multiplier = 1.0
        
        base_hours = 1.0 * time_multiplier  # 1 hour per page as base
        
        # Adjust based on scene characteristics
        cast = scene.get('Cast', [])
        cast_size = len(cast) if isinstance(cast, list) else 1
        if cast_size > 3:
            base_hours *= 1.5  # More cast = more time
        
        # Adjust based on INT/EXT
        if scene.get('INT_EXT') == 'EXT':
            base_hours *= 1.3  # Exterior scenes take longer
        
        return base_hours

    def _parse_page_count(self, page_count_str: str) -> float:
        """Parse page count strings like '1 6/8', '3/8', '2 1/8' into decimal multipliers"""
        try:
            # Handle formats like "1 6/8", "3/8", "2"
            if '/' in page_count_str:
                parts = page_count_str.strip().split()
                if len(parts) == 2:  # "1 6/8"
                    whole = int(parts[0])
                    frac_parts = parts[1].split('/')
                    fraction = int(frac_parts[0]) / int(frac_parts[1])
                    return whole + fraction
                else:  # "6/8"
                    frac_parts = page_count_str.split('/')
                    return int(frac_parts[0]) / int(frac_parts[1])
            else:  # "2"
                return float(page_count_str.strip())
        except:
            return 1.0  # Default if parsing fails

    def _add_schedule_day(self, schedule: List[Dict], shooting_day_idx: int, 
                        daily_scenes: List[Dict], cluster: 'LocationCluster', 
                        scheduled_scenes: set):
        """Helper method to add a day to the schedule"""
        if shooting_day_idx >= len(self.calendar.shooting_days):
            return  # Can't add day beyond calendar
        
        shooting_date = self.calendar.shooting_days[shooting_day_idx]
        
        # Mark scenes as scheduled
        for scene in daily_scenes:
            scheduled_scenes.add(scene['Scene_Number'])
        
        # Calculate actual hours for this day
        actual_hours = sum(self._get_scene_hours(scene) for scene in daily_scenes)
        
        schedule.append({
            'day': shooting_day_idx + 1,
            'date': shooting_date.strftime("%Y-%m-%d"),
            'location': cluster.location,
            'location_name': self._extract_location_name(cluster.location),
            'scenes': daily_scenes,
            'scene_count': len(daily_scenes),
            'location_moves': 0,  # Single location per day
            'estimated_hours': round(actual_hours, 1)
        })

    def _extract_location_name(self, geographic_location: str) -> str:
        """Extract location name from geographic address"""
        # Try to find location name from original stripboard
        for scene in self.stripboard:
            if scene.get('Geographic_Location') == geographic_location:
                return scene.get('Location_Name', 'Unknown Location')
        return 'Unknown Location'

    def _split_scene_into_parts(self, scene: Dict, total_hours: float, max_daily_hours: float) -> List[Dict]:
        """Split a complex scene into multiple parts following Option C (Hybrid) approach"""
        
        # Calculate how many parts we need
        total_parts = int((total_hours + max_daily_hours - 0.1) / max_daily_hours)  # Ceiling division
        
        # Calculate hours per part
        hours_per_part = total_hours / total_parts
        
        scene_parts = []
        remaining_hours = total_hours
        
        for part_num in range(1, total_parts + 1):
            # Calculate hours for this part
            if part_num == total_parts:
                # Last part gets all remaining hours
                part_hours = remaining_hours
            else:
                # Regular part gets calculated portion, but cap at max_daily_hours
                part_hours = min(hours_per_part, max_daily_hours)
            
            # Create scene part using Option C (Hybrid) approach
            scene_part = scene.copy()  # Start with original scene data
            
            # Option C: Add hybrid fields
            scene_part['Display_Name'] = f"Scene {scene['Scene_Number']} (Part {part_num} of {total_parts})"
            scene_part['split_info'] = {
                'is_split': True,
                'part': part_num,
                'total_parts': total_parts,
                'part_hours': round(part_hours, 1),
                'original_total_hours': round(total_hours, 1)
            }
            
            scene_parts.append(scene_part)
            remaining_hours -= part_hours
        
        print(f"DEBUG: Split Scene {scene['Scene_Number']} ({total_hours:.1f}h) into {total_parts} parts:")
        for part in scene_parts:
            split_info = part['split_info']
            print(f"  - {part['Display_Name']}: {split_info['part_hours']}h")
        
        return scene_parts


    def _calculate_metrics(self, schedule: List[Dict]) -> Dict[str, Any]:
        """Calculate metrics including REAL hard conflicts from constraint violations"""
        if not schedule:
            return {
                'total_shooting_days': 0,
                'total_scenes': 0,
                'total_location_moves': 0,
                'total_geographic_moves': 0,
                'theoretical_minimum_moves': 0,
                'efficiency_ratio': 1.0,
                'avg_locations_per_day': 0,
                'constraint_satisfaction_rate': 0,
                'hard_conflicts': 0,  # This was always 0 before
                'soft_conflicts': 0
            }
        
        # Count unique scenes (prevent duplication counting)
        unique_scenes = set()
        for day in schedule:
            for scene in day['scenes']:
                unique_scenes.add(scene['Scene_Number'])
        
        total_moves = sum(day.get('location_moves', 0) for day in schedule)
        n_unique_locations = len(self.cluster_manager.clusters)
        
        # NEW: Calculate geographic location moves (crew moves between locations)
        total_geographic_moves = self._calculate_geographic_moves(schedule)
        
        # FIXED: Correct theoretical minimum moves
        theoretical_minimum_moves = max(0, n_unique_locations - 1)
        
        # Calculate efficiency - fewer geographic moves is better
        if total_geographic_moves == 0:
            efficiency_ratio = 1.0  # Perfect efficiency
        else:
            efficiency_ratio = theoretical_minimum_moves / max(total_geographic_moves, 1)
        
        # Calculate locations per day
        locations_per_day = []
        for day in schedule:
            if 'location' in day:
                locations_per_day.append(1)  # Single location per day
            else:
                locations_per_day.append(0)
        
        avg_locations_per_day = sum(locations_per_day) / len(locations_per_day) if locations_per_day else 0
        
        # FIX: Calculate REAL hard conflicts by recreating the GA's violation detection
        hard_conflicts = self._calculate_real_hard_conflicts_from_schedule(schedule)
        
        # Calculate constraint satisfaction based on actual conflicts
        total_constraints = len(self.constraints)
        if total_constraints > 0:
            satisfaction_rate = max(0.0, 1.0 - (hard_conflicts / total_constraints))
        else:
            satisfaction_rate = 1.0
        
        return {
            'total_shooting_days': len(schedule),
            'total_scenes': len(unique_scenes),
            'total_location_moves': total_moves,
            'total_geographic_moves': total_geographic_moves,
            'theoretical_minimum_moves': theoretical_minimum_moves,
            'efficiency_ratio': round(efficiency_ratio, 3),
            'avg_locations_per_day': round(avg_locations_per_day, 2),
            'constraint_satisfaction_rate': round(satisfaction_rate, 2),
            'hard_conflicts': hard_conflicts,  # NOW REAL VALUES!
            'soft_conflicts': 0  # Will implement in later steps
        }

    def _calculate_real_hard_conflicts_from_schedule(self, schedule: List[Dict]) -> int:
        """NEW: Calculate hard conflicts by converting schedule back to GA format and checking violations"""
        try:
            # Convert schedule back to GA individual format
            sequence, day_assignments = self._schedule_to_ga_format(schedule)
            
            # Use the same violation detection as fitness function
            actor_violations = self._check_complete_actor_violations(sequence, day_assignments)
            
            # Add other violation types as we implement them in later steps
            # director_violations = self._check_director_mandate_violations(sequence, day_assignments)
            # location_violations = self._check_location_availability_violations(sequence, day_assignments)
            
            total_hard_conflicts = actor_violations
            print(f"DEBUG: Metrics - Found {total_hard_conflicts} hard conflicts ({actor_violations} actor violations)")
            
            return total_hard_conflicts
            
        except Exception as e:
            print(f"DEBUG: Error calculating real hard conflicts: {e}")
            return 0    

    def _schedule_to_ga_format(self, schedule: List[Dict]) -> Tuple[List[int], List[int]]:
        """NEW: Convert final schedule back to GA individual format for violation checking"""
        sequence = []
        day_assignments = []
        
        # Map locations to cluster indices
        location_to_cluster = {}
        for i, cluster in enumerate(self.cluster_manager.clusters):
            location_to_cluster[cluster.location] = i
        
        current_cluster_idx = None
        current_start_day = None
        
        for day_idx, day in enumerate(schedule):
            location = day.get('location', '')
            cluster_idx = location_to_cluster.get(location)
            
            if cluster_idx is not None:
                if current_cluster_idx != cluster_idx:
                    # New cluster started
                    if current_cluster_idx is not None:
                        # Save previous cluster
                        sequence.append(current_cluster_idx)
                        day_assignments.append(current_start_day)
                    
                    current_cluster_idx = cluster_idx
                    current_start_day = day_idx
        
        # Add final cluster
        if current_cluster_idx is not None:
            sequence.append(current_cluster_idx)
            day_assignments.append(current_start_day)
        
        return sequence, day_assignments

    def _calculate_day_hours(self, daily_scenes: List[Dict]) -> float:
        """Calculate total estimated hours for a day using real scene time estimates"""
        total_hours = 0.0
        
        # Get scene time estimates from constraints
        scene_estimates = self._get_scene_time_estimates()
        
        for scene in daily_scenes:
            scene_number = scene['Scene_Number']
            
            # Look up actual time estimate
            estimated_hours = scene_estimates.get(scene_number, 1.5)  # Default 1.5 hours if not found
            total_hours += estimated_hours
        
        return round(total_hours, 1)
    
    def _get_scene_time_estimates(self) -> Dict[str, float]:
        """Extract scene time estimates from operational data - WITH DEBUG LOGGING"""
        scene_estimates = {}
        
        try:
            print(f"DEBUG: constraints_raw top-level keys: {list(self.constraints_raw.keys())}")
            
            # Check if operational_data exists
            if 'operational_data' in self.constraints_raw:
                operational_data = self.constraints_raw['operational_data']
                print(f"DEBUG: operational_data keys: {list(operational_data.keys())}")
                
                # Check time_estimates structure
                if 'time_estimates' in operational_data:
                    time_estimates = operational_data['time_estimates']
                    print(f"DEBUG: time_estimates keys: {list(time_estimates.keys()) if isinstance(time_estimates, dict) else 'Not a dict'}")
                    
                    # Check scene_estimates
                    if 'scene_estimates' in time_estimates:
                        scene_estimates_data = time_estimates['scene_estimates']
                        print(f"DEBUG: scene_estimates_data type: {type(scene_estimates_data)}")
                        print(f"DEBUG: scene_estimates_data length: {len(scene_estimates_data) if hasattr(scene_estimates_data, '__len__') else 'No length'}")
                        
                        if isinstance(scene_estimates_data, list) and len(scene_estimates_data) > 0:
                            print(f"DEBUG: First scene estimate sample: {scene_estimates_data[0]}")
                            
                            '''for scene_est in scene_estimates_data:
                                if isinstance(scene_est, dict):
                                    scene_number = scene_est.get('Scene_Number')
                                    estimated_hours = scene_est.get('Estimated_Time_Hours', 1.5)
                                    
                                    if scene_number:
                                        scene_estimates[str(scene_number)] = float(estimated_hours)
                                        print(f"DEBUG: Loaded scene {scene_number}: {estimated_hours} hours") '''
                            for scene_est in scene_estimates_data:
                                if isinstance(scene_est, dict):
                                    # FIX 1: Handle Unicode BOM in key names
                                    scene_number = None
                                    estimated_hours = 1.5
                                    
                                    # Try multiple key variations for scene number
                                    for key in scene_est.keys():
                                        if 'Scene_Number' in key:  # Handles both 'Scene_Number' and '\ufeffScene_Number'
                                            scene_number = scene_est[key]
                                            break
                                    
                                    # Try multiple key variations for estimated hours
                                    for key in scene_est.keys():
                                        if 'Estimated_Time_Hours' in key:
                                            estimated_hours = scene_est[key]
                                            break
                                    
                                    if scene_number:
                                        try:
                                            # FIX 2: Convert string to float
                                            hours_float = float(estimated_hours) if estimated_hours else 1.5
                                            scene_estimates[str(scene_number)] = hours_float
                                            print(f"DEBUG: Loaded scene {scene_number}: {hours_float} hours")
                                        except (ValueError, TypeError):
                                            print(f"DEBUG: Failed to convert hours for scene {scene_number}: {estimated_hours}")
                                            scene_estimates[str(scene_number)] = 1.5                                        
                        else:
                            print(f"DEBUG: scene_estimates_data is empty or not a list")
                    else:
                        print(f"DEBUG: 'scene_estimates' key not found in time_estimates")
                else:
                    print(f"DEBUG: 'time_estimates' key not found in operational_data")
            else:
                print(f"DEBUG: 'operational_data' key not found in constraints_raw")
                
                # ALTERNATIVE PATH: Check if scene estimates are elsewhere in the structure
                print(f"DEBUG: Searching for scene estimates in alternative locations...")
                
                # Check if scene estimates are directly in constraints
                for key, value in self.constraints_raw.items():
                    if isinstance(value, dict):
                        print(f"DEBUG: Checking {key}: {list(value.keys()) if isinstance(value, dict) else type(value)}")
                        
                        # Look for scenes or time estimates in any top-level constraint group
                        if 'scenes' in value or 'time_estimates' in value or 'scene_estimates' in value:
                            print(f"DEBUG: Found potential scene data in {key}")
            
            print(f"DEBUG: Final loaded scene estimates count: {len(scene_estimates)}")
            
        except Exception as e:
            print(f"DEBUG: Error in scene estimates extraction: {e}")
            import traceback
            traceback.print_exc()
        
        return scene_estimates

    def _calculate_geographic_moves(self, schedule: List[Dict]) -> int:
        """Calculate total geographic location moves (crew relocations)"""
        if len(schedule) <= 1:
            return 0
        
        geographic_moves = 0
        previous_location = None
        
        for day in schedule:
            current_location = day.get('location')
            
            if previous_location is not None and current_location != previous_location:
                geographic_moves += 1
            
            previous_location = current_location
        
        return geographic_moves    

    def _build_schedule_summary(self, schedule: List[Dict]) -> Dict[str, Any]:
        """Build schedule summary matching desired output format"""
        if not schedule:
            return {
                'total_shooting_days': 0,
                'total_scenes': 0,
                'total_locations': 0,
                'total_location_moves': 0
            }
        
        # Count unique scenes (no duplicates)
        unique_scenes = set()
        unique_locations = set()
        total_moves = 0
        
        for day in schedule:
            # Count unique scenes
            for scene in day['scenes']:
                unique_scenes.add(scene['Scene_Number'])
            
            # Count unique locations
            if 'location' in day:
                unique_locations.add(day['location'])
            
            # Sum location moves
            total_moves += day.get('location_moves', 0)
        
        return {
            'total_shooting_days': len(schedule),
            'total_scenes': len(unique_scenes),
            'total_locations': len(unique_locations),
            'total_location_moves': total_moves
        }

@app.post("/optimize/schedule", response_model=ScheduleResponse)
async def optimize_schedule(request: ScheduleRequest):
    """
    Location-First Film Schedule Optimization
    Prioritizes geographic location clustering while respecting constraints
    """
    try:
        optimizer = ScheduleOptimizer(request)
        result = optimizer.optimize()
        
        return ScheduleResponse(
            schedule=result['schedule'],
            summary=result['summary'],                     # NEW
            missing_scenes=result['missing_scenes'],       # NEW
            conflicts=result['conflicts'],
            metrics=result['metrics'],
            fitness_score=result['fitness_score'],
            processing_time_seconds=result['processing_time_seconds']
        )
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "version": "4.0.0", "approach": "location-first", "penalty_system": "graduated"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
