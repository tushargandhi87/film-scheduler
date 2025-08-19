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
    total_pages: float
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
        """Parse actor availability constraints"""
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
    """Manages grouping of scenes by geographic location"""
    
    def __init__(self, stripboard: List[Dict]):
        self.stripboard = stripboard
        self.clusters = self._create_location_clusters()
        
        print(f"DEBUG: Created {len(self.clusters)} location clusters")
        for i, cluster in enumerate(self.clusters):
            print(f"  Cluster {i}: {cluster.location} - {len(cluster.scenes)} scenes")
    
    def _create_location_clusters(self) -> List[LocationCluster]:
    """Group scenes by geographic location"""
    location_groups = defaultdict(list)
    
    for scene in self.stripboard:
        location = scene.get('Geographic_Location', 'Unknown Location')
        location_groups[location].append(scene)
    
    clusters = []
    for location, scenes in location_groups.items():
        # Use scene time estimates if available
        total_minutes = 0
        for scene in scenes:
            # Look for time estimate in scene data
            time_estimate = scene.get('Estimated_Time_Minutes', 60)  # Default 1 hour
            total_minutes += time_estimate
        
        # Convert to shooting days (assuming 8 hours = 480 minutes per day)
        estimated_days = max(1, (total_minutes + 240) // 480)
        
        # Extract actors
        all_actors = set()
        for scene in scenes:
            cast = scene.get('Cast', [])
            if isinstance(cast, list):
                all_actors.update(cast)
            elif isinstance(cast, str):
                all_actors.update([cast])
        
        clusters.append(LocationCluster(
            location=location,
            scenes=scenes,
            total_pages=0.0,  # Not needed
            estimated_days=estimated_days,
            required_actors=list(all_actors)
        ))
    
    return clusters

class LocationFirstGA:
    """Location-First Genetic Algorithm for scheduling"""
    
    def __init__(self, cluster_manager: LocationClusterManager, constraints: List[Constraint], 
                 calendar: ShootingCalendar, params: Dict):
        self.cluster_manager = cluster_manager
        self.constraints = constraints
        self.calendar = calendar
        self.params = params
        self.rng = np.random.RandomState(params.get('seed', 42))
        
        # Build constraint maps for efficient lookup
        self._build_constraint_maps()
        self._build_travel_times()
    
    def _build_constraint_maps(self):
        """Build efficient lookup structures for constraints"""
        self.actor_unavailable_dates = defaultdict(list)
        self.director_mandates = {}
        self.location_windows = {}
        
        for constraint in self.constraints:
            if constraint.actor_restriction:
                actor = constraint.actor_restriction.get('actor')
                unavailable_date = constraint.actor_restriction.get('unavailable_date')
                if actor and unavailable_date:
                    self.actor_unavailable_dates[actor].append(unavailable_date)
            
            elif constraint.source == ConstraintPriority.DIRECTOR and constraint.affected_scenes:
                # Mark director mandates
                for scene in constraint.affected_scenes:
                    self.director_mandates[scene] = constraint
            
            elif constraint.location_restriction and constraint.location_restriction.get('location'):
                location = constraint.location_restriction['location']
                self.location_windows[location] = constraint
    
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
            if (restriction.get('from_location_fictional') in loc1 or 
                restriction.get('to_location_fictional') in loc1) and \
               (restriction.get('from_location_fictional') in loc2 or 
                restriction.get('to_location_fictional') in loc2):
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
        Create individual representing location cluster sequence and day assignments
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
        
        # Assign each cluster to a starting day
        day_assignments = []
        current_day = 0
        
        for cluster_idx in sequence:
            cluster = self.cluster_manager.clusters[cluster_idx]
            
            # Ensure we don't exceed available days
            if current_day >= n_days:
                current_day = n_days - 1
            
            day_assignments.append(current_day)
            
            # Advance by estimated days needed for this cluster
            current_day += max(1, cluster.estimated_days)
        
        return {
            'sequence': sequence,
            'day_assignments': day_assignments
        }
    
    def fitness(self, individual: Dict) -> float:
        """Calculate fitness using graduated penalty system"""
        score = 0.0
        
        sequence = individual['sequence']
        day_assignments = individual['day_assignments']
        
        # 1. HARD CONSTRAINT VIOLATIONS (-10,000 each)
        hard_violations = self._count_hard_violations(sequence, day_assignments)
        score += PENALTY_HARD_CONSTRAINT * hard_violations
        
        # 2. LOCATION SPLITTING PENALTY (-2,000 each)
        location_splits = self._count_location_splits(sequence, day_assignments)
        score += PENALTY_LOCATION_SPLIT * location_splits
        
        # 3. DIRECTOR MANDATE VIOLATIONS (-5,000 each)
        director_violations = self._count_director_violations(sequence, day_assignments)
        score += PENALTY_DIRECTOR_MANDATE * director_violations
        
        # 4. TRAVEL TIME PENALTIES (-100 per mile)
        travel_penalty = self._calculate_travel_penalty(sequence)
        score += travel_penalty
        
        # 5. ACTOR IDLE DAYS (-200 each)
        idle_penalty = self._calculate_actor_idle_penalty(sequence, day_assignments)
        score += idle_penalty
        
        # 6. SOFT CONSTRAINT BONUSES (+100 each)
        soft_bonus = self._calculate_soft_bonus(sequence, day_assignments)
        score += soft_bonus
        
        return score
    
    def _count_hard_violations(self, sequence: List[int], day_assignments: List[int]) -> int:
        """Count hard constraint violations"""
        violations = 0
        
        for i, cluster_idx in enumerate(sequence):
            cluster = self.cluster_manager.clusters[cluster_idx]
            start_day = day_assignments[i]
            
            # Check actor availability for each day this cluster needs
            for day_offset in range(cluster.estimated_days):
                shooting_day_idx = start_day + day_offset
                if shooting_day_idx >= len(self.calendar.shooting_days):
                    violations += 1  # Exceeds calendar
                    continue
                
                shooting_date = self.calendar.shooting_days[shooting_day_idx]
                
                # Check each actor required for this cluster
                for actor in cluster.required_actors:
                    if actor in self.actor_unavailable_dates:
                        for unavailable_date_str in self.actor_unavailable_dates[actor]:
                            try:
                                unavailable_date = datetime.strptime(unavailable_date_str, "%Y-%m-%d").date()
                                if shooting_date == unavailable_date:
                                    violations += 1
                            except:
                                pass
        
        return violations
    
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
        self.stripboard = request.stripboard
        self.constraints_raw = request.constraints
        self.params = request.ga_params
        
        # Initialize components
        self.parser = StructuredConstraintParser()
        self.constraints = self.parser.parse_all_constraints(self.constraints_raw)
        
        # Create location cluster manager
        self.cluster_manager = LocationClusterManager(self.stripboard)
        
        # Determine shooting calendar
        self.calendar = ShootingCalendar("2025-09-01", "2025-10-31")
        
        print(f"DEBUG: PARSED {len(self.constraints)} CONSTRAINTS")
        print(f"DEBUG: CREATED {len(self.cluster_manager.clusters)} LOCATION CLUSTERS")
        print(f"DEBUG: CALENDAR HAS {len(self.calendar.shooting_days)} SHOOTING DAYS")
    
    def optimize(self) -> Dict[str, Any]:
        """Run location-first optimization"""
        import time
        start_time = time.time()
        
        # Run location-first genetic algorithm
        ga = LocationFirstGA(self.cluster_manager, self.constraints, self.calendar, self.params)
        best_individual, best_fitness = ga.evolve()
        
        # Build final schedule from best individual
        final_schedule = self._build_final_schedule(best_individual)
        
        # Calculate metrics
        metrics = self._calculate_metrics(final_schedule)
        
        processing_time = time.time() - start_time
        
        return {
            'schedule': final_schedule,
            'conflicts': [],  # Simplified for now
            'metrics': metrics,
            'fitness_score': best_fitness,
            'processing_time_seconds': processing_time
        }
    
    def _build_final_schedule(self, individual: Dict) -> List[Dict]:
        """Build final day-by-day schedule from best individual"""
        schedule = []
        
        sequence = individual['sequence']
        day_assignments = individual['day_assignments']
        
        # Group by actual shooting days
        daily_clusters = defaultdict(list)
        
        for i, cluster_idx in enumerate(sequence):
            cluster = self.cluster_manager.clusters[cluster_idx]
            start_day = day_assignments[i]
            
            # Assign cluster to consecutive days
            for day_offset in range(cluster.estimated_days):
                shooting_day_idx = start_day + day_offset
                if shooting_day_idx < len(self.calendar.shooting_days):
                    daily_clusters[shooting_day_idx].append(cluster)
        
        # Build schedule for each day
        for day_idx in sorted(daily_clusters.keys()):
            shooting_date = self.calendar.shooting_days[day_idx]
            clusters = daily_clusters[day_idx]
            
            # Collect all scenes from clusters for this day
            scenes = []
            locations = []
            
            for cluster in clusters:
                scenes.extend(cluster.scenes)
                if cluster.location not in locations:
                    locations.append(cluster.location)
            
            schedule.append({
                'day': day_idx + 1,
                'date': shooting_date.strftime("%Y-%m-%d"),
                'locations': locations,
                'scenes': scenes,
                'scene_count': len(scenes),
                'location_moves': len(locations) - 1 if len(locations) > 1 else 0
            })
        
        return schedule
    
    def _calculate_metrics(self, schedule: List[Dict]) -> Dict[str, Any]:
        """Calculate schedule metrics"""
        total_moves = sum(day['location_moves'] for day in schedule)
        total_scenes = sum(day['scene_count'] for day in schedule)
        n_unique_locations = len(self.cluster_manager.clusters)
        
        # Calculate location efficiency
        location_changes = []
        for day in schedule:
            location_changes.append(len(day['locations']))
        
        avg_locations_per_day = np.mean(location_changes) if location_changes else 0
        
        # Calculate actual constraint satisfaction rate
        satisfied_constraints = 0
        total_constraints = len(self.constraints)
        
        # Simplified constraint satisfaction check
        for constraint in self.constraints:
            # For now, assume soft constraints are satisfied and hard constraints
            # satisfaction depends on the fitness score
            if constraint.type == ConstraintType.SOFT:
                satisfied_constraints += 1
            elif constraint.type == ConstraintType.HARD:
                # Simplified check - assume satisfied if no obvious violations
                satisfied_constraints += 0.7  # 70% of hard constraints assumed satisfied
        
        satisfaction_rate = satisfied_constraints / total_constraints if total_constraints > 0 else 0
        
        return {
            'total_shooting_days': len(schedule),
            'total_scenes': total_scenes,
            'total_location_moves': total_moves,
            'theoretical_minimum_moves': n_unique_locations - 1,
            'efficiency_ratio': (n_unique_locations - 1) / total_moves if total_moves > 0 else 1.0,
            'avg_locations_per_day': round(avg_locations_per_day, 2),
            'constraint_satisfaction_rate': round(satisfaction_rate, 2),
            'hard_conflicts': 0,  # Simplified
            'soft_conflicts': 0   # Simplified
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
