"""
Advanced Film Production Schedule Optimizer
3-Phase Genetic Algorithm with Hierarchical Constraint Satisfaction
Updated to work with structured JSON from n8n workflow
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

app = FastAPI(title="Advanced Film Schedule Optimizer v3.0")

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
INFINITY_PENALTY = -999999

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
class Conflict:
    """Conflict representation for AD review"""
    scene_id: str
    severity: str  # HIGH, MEDIUM, LOW
    constraint_1: str
    constraint_2: str
    suggested_resolution: str

class ScheduleRequest(BaseModel):
    """Updated request model for structured constraints from n8n"""
    stripboard: List[Dict[str, Any]]
    constraints: Dict[str, Any]  # Structured constraints from n8n grouping
    ga_params: Optional[Dict[str, Any]] = {
        "phase1_population": 50,
        "phase1_generations": 200,
        "phase2_population": 100,
        "phase2_generations": 500,
        "phase3_population": 100,
        "phase3_generations": 500,
        "mutation_rate": 0.02,
        "crossover_rate": 0.85,
        "seed": 42,
        "conflict_tolerance": 0.1
    }

class ScheduleResponse(BaseModel):
    """Response model with optimized schedule"""
    schedule: List[Dict[str, Any]]  # Day-by-day schedule
    conflicts: List[Dict[str, Any]]  # Conflicts for AD review
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
            # Parse each constraint group with error handling
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
        
        # Add defensive programming to handle any structure
        try:
            if 'actors' in people_data:
                actors_data = people_data['actors']
                
                # Handle multiple possible nested structures
                if isinstance(actors_data, dict) and 'actors' in actors_data:
                    actors_info = actors_data['actors']
                elif isinstance(actors_data, dict):
                    actors_info = actors_data
                else:
                    print(f"DEBUG: Unexpected actors_data type: {type(actors_data)}")
                    return constraints
                
                if not isinstance(actors_info, dict):
                    print(f"DEBUG: actors_info is not dict: {type(actors_info)}")
                    return constraints
                
                for actor_name, actor_info in actors_info.items():
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
        
        return constraints
    
    def _parse_location_constraints(self, location_data: Dict) -> List[Constraint]:
        """Parse location availability and travel time constraints"""
        constraints = []
        
        # Add defensive programming
        try:
            # Parse location availability
            if 'locations' in location_data:
                locations_info = location_data['locations']
                
                # Handle nested structure safely
                if isinstance(locations_info, dict) and 'locations' in locations_info:
                    locations_dict = locations_info['locations']
                elif isinstance(locations_info, dict):
                    locations_dict = locations_info
                else:
                    print(f"DEBUG: Unexpected locations_info type: {type(locations_info)}")
                    locations_dict = {}
                
                if isinstance(locations_dict, dict):
                    for location_name, location_info in locations_dict.items():
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
        
            # Parse travel times safely
            if 'travel_times' in location_data:
                travel_data = location_data['travel_times']
                if isinstance(travel_data, dict) and 'travel_times' in travel_data:
                    travel_times = travel_data['travel_times']
                else:
                    travel_times = travel_data
                
                # Handle new array structure
                if isinstance(travel_times, list):
                    for travel_info in travel_times:
                        if isinstance(travel_info, dict):
                            route_key = f"{travel_info.get('from_location_fictional', '')}_to_{travel_info.get('to_location_fictional', '')}"
                            constraints.append(Constraint(
                                source=ConstraintPriority.LOCATION,
                                type=ConstraintType.SOFT,
                                description=f"Travel time {route_key}: {travel_info.get('estimated_travel_time_minutes', 0)} minutes",
                                affected_scenes=[],
                                location_restriction={
                                    'from_location': travel_info.get('from_location_fictional', ''),
                                    'to_location': travel_info.get('to_location_fictional', ''),
                                    'travel_time_minutes': travel_info.get('estimated_travel_time_minutes', 0),
                                    'notes': travel_info.get('notes', '')
                                }
                            ))
                elif isinstance(travel_times, dict):
                    # Handle old dictionary structure (fallback)
                    for route, time_str in travel_times.items():
                        constraints.append(Constraint(
                            source=ConstraintPriority.LOCATION,
                            type=ConstraintType.SOFT,
                            description=f"Travel time {route}: {time_str}",
                            affected_scenes=[],
                            location_restriction={
                                'route': route,
                                'travel_time': time_str
                            }
                        ))
        
        except Exception as e:
            print(f"DEBUG: Error parsing location constraints: {e}")
        
        return constraints
    
    def _parse_technical_constraints(self, technical_data: Dict) -> List[Constraint]:
        """Parse equipment and special requirements"""
        constraints = []
        
        # Parse equipment constraints
        if 'equipment' in technical_data:
            equipment_data = technical_data['equipment']
            
            # Handle nested structure
            if 'special_equipment' in equipment_data:
                equipment_dict = equipment_data['special_equipment']
            else:
                equipment_dict = equipment_data
            
            for equipment_name, equipment_info in equipment_dict.items():
                constraint_level = equipment_info.get('constraint_level', 'Hard')
                constraint_type = ConstraintType.HARD if constraint_level == 'Hard' else ConstraintType.SOFT
                
                constraints.append(Constraint(
                    source=ConstraintPriority.EQUIPMENT,
                    type=constraint_type,
                    description=f"{equipment_name}: {equipment_info.get('notes', '')}",
                    affected_scenes=[],
                    location_restriction={  # Using location_restriction for equipment
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
            
            # Handle nested structure
            if 'scene_requirements' in special_data:
                scene_reqs = special_data['scene_requirements']
            else:
                scene_reqs = special_data
            
            for req_name, req_info in scene_reqs.items():
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
        
        return constraints
    
    def _parse_creative_constraints(self, creative_data: Dict) -> List[Constraint]:
        """Parse director and DOP constraints"""
        constraints = []
        
        try:
            # Parse director constraints
            if 'director_notes' in creative_data:
                director_data = creative_data['director_notes']
                
                # Handle case where director_notes is still a markdown-wrapped string
                if isinstance(director_data, str):
                    # Clean markdown and parse JSON
                    clean_json = director_data.replace('```json\n', '').replace('```', '').strip()
                    try:
                        director_data = json.loads(clean_json)
                    except json.JSONDecodeError:
                        print(f"DEBUG: Failed to parse director_notes JSON: {clean_json[:100]}...")
                        return constraints
                
                # Handle nested structure
                if isinstance(director_data, dict) and 'director_constraints' in director_data:
                    director_constraints = director_data['director_constraints']
                elif isinstance(director_data, list):
                    director_constraints = director_data
                else:
                    print(f"DEBUG: Unexpected director_data structure: {type(director_data)}")
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
                
                # Handle case where dop_priorities is still a markdown-wrapped string
                if isinstance(dop_data, str):
                    # Clean markdown and parse JSON
                    clean_json = dop_data.replace('```json\n', '').replace('```', '').strip()
                    try:
                        dop_data = json.loads(clean_json)
                    except json.JSONDecodeError:
                        print(f"DEBUG: Failed to parse dop_priorities JSON: {clean_json[:100]}...")
                        return constraints
                
                # Handle nested structure
                if isinstance(dop_data, dict) and 'dop_priorities' in dop_data:
                    dop_constraints = dop_data['dop_priorities']
                elif isinstance(dop_data, list):
                    dop_constraints = dop_data
                else:
                    print(f"DEBUG: Unexpected dop_data structure: {type(dop_data)}")
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
        
        # Parse production rules
        if 'production_rules' in operational_data:
            prod_data = operational_data['production_rules']
            
            # Handle nested structure
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
    
    def parse_date_restriction(self, restriction: str) -> List[int]:
        """Parse date restriction string into day indices"""
        indices = []
        # Parse various date formats
        # "Sept 1-6", "Weekends only", etc.
        # Returns list of valid shooting day indices
        return indices

class ConflictDetector:
    """Detects and reports scheduling conflicts"""
    
    def __init__(self, constraints: List[Constraint], stripboard: List[Dict]):
        self.constraints = constraints
        self.stripboard = stripboard
        self.conflicts = []
    
    def detect_all_conflicts(self) -> List[Conflict]:
        """Detect all conflicts in the constraint set"""
        self.conflicts = []
        
        # Check actor vs location conflicts
        self._check_actor_location_conflicts()
        
        # Check director mandate conflicts
        self._check_director_mandate_conflicts()
        
        # Check equipment availability conflicts
        self._check_equipment_conflicts()
        
        return self.conflicts
    
    def _check_actor_location_conflicts(self):
        """Check if actor unavailable when location is available"""
        actor_constraints = [c for c in self.constraints if c.actor_restriction]
        location_constraints = [c for c in self.constraints if c.location_restriction]
        
        for actor_c in actor_constraints:
            for location_c in location_constraints:
                # Check if there's a date overlap conflict
                if self._dates_conflict(actor_c, location_c):
                    # Find affected scenes
                    affected_scenes = self._find_affected_scenes(
                        actor_c.actor_restriction.get('actor'),
                        location_c.location_restriction.get('location')
                    )
                    
                    if affected_scenes:
                        self.conflicts.append(Conflict(
                            scene_id=affected_scenes[0],
                            severity="HIGH",
                            constraint_1=actor_c.description,
                            constraint_2=location_c.description,
                            suggested_resolution="Reschedule scenes or use different location/actor double"
                        ))
    
    def _check_director_mandate_conflicts(self):
        """Check if director mandates conflict with other constraints"""
        director_constraints = [c for c in self.constraints if c.source == ConstraintPriority.DIRECTOR]
        
        for dir_c in director_constraints:
            for other_c in self.constraints:
                if other_c.source != ConstraintPriority.DIRECTOR and self._constraints_conflict(dir_c, other_c):
                    self.conflicts.append(Conflict(
                        scene_id=dir_c.affected_scenes[0] if dir_c.affected_scenes else "Multiple",
                        severity="HIGH",
                        constraint_1=dir_c.description,
                        constraint_2=other_c.description,
                        suggested_resolution="Director mandate takes priority - AD manual review required"
                    ))
    
    def _check_equipment_conflicts(self):
        """Check equipment rental period conflicts"""
        equipment_constraints = [c for c in self.constraints if c.source == ConstraintPriority.EQUIPMENT]
        
        for eq_c in equipment_constraints:
            # Check if scenes requiring this equipment can fit in rental period
            pass
    
    def _dates_conflict(self, c1: Constraint, c2: Constraint) -> bool:
        """Check if two constraints have conflicting dates"""
        # Implement date conflict logic
        return False
    
    def _constraints_conflict(self, c1: Constraint, c2: Constraint) -> bool:
        """Check if two constraints conflict"""
        # Implement general conflict logic
        return False
    
    def _find_affected_scenes(self, actor: str, location: str) -> List[str]:
        """Find scenes with specific actor at specific location"""
        affected = []
        for scene in self.stripboard:
            if actor in scene.get('Cast', []) and location in scene.get('Location_Name', ''):
                affected.append(scene['Scene_Number'])
        return affected

class Phase1GA:
    """Phase 1: Temporal Scheduling - Assign scenes to dates"""
    
    def __init__(self, stripboard: List[Dict], constraints: List[Constraint], 
                 calendar: ShootingCalendar, params: Dict):
        self.stripboard = stripboard
        self.constraints = constraints
        self.calendar = calendar
        self.params = params
        self.rng = np.random.RandomState(params.get('seed', 42))
        
        # Build constraint lookup maps
        self._build_constraint_maps()
    
    def _build_constraint_maps(self):
        """Build efficient lookup structures for constraints"""
        self.actor_unavailable = defaultdict(list)  # actor -> list of unavailable days
        self.location_windows = defaultdict(tuple)   # location -> (start_day, end_day)
        self.director_mandates = {}                  # scene -> required_day
        
        for constraint in self.constraints:
            if constraint.actor_restriction:
                actor = constraint.actor_restriction.get('actor')
                self.actor_unavailable[actor].append(constraint.actor_restriction)
            elif constraint.location_restriction:
                location = constraint.location_restriction.get('location')
                self.location_windows[location] = constraint.location_restriction
            elif constraint.source == ConstraintPriority.DIRECTOR and constraint.date_restriction:
                for scene in constraint.affected_scenes:
                    self.director_mandates[scene] = constraint.date_restriction.get('day', 1)
    
    def create_individual(self) -> np.ndarray:
        """Create individual: scene -> shooting day assignment"""
        n_scenes = len(self.stripboard)
        n_days = len(self.calendar.shooting_days)
        
        # Random assignment with basic constraint awareness
        individual = self.rng.randint(0, n_days, n_scenes)
        
        # Apply director mandates
        for scene_idx, scene in enumerate(self.stripboard):
            scene_num = scene['Scene_Number']
            if scene_num in self.director_mandates:
                individual[scene_idx] = self.director_mandates[scene_num] - 1  # Convert to 0-index
        
        return individual
    
    def fitness(self, individual: np.ndarray) -> float:
        """Calculate fitness for temporal assignment"""
        score = 0.0
        
        # 1. Hard constraint violations
        hard_violations = self._count_hard_violations(individual)
        print(f"DEBUG Phase1: Hard violations = {hard_violations}")

        if hard_violations > 0:
            score += INFINITY_PENALTY * hard_violations
        
        # 2. Soft constraint satisfaction
        soft_score = self._evaluate_soft_constraints(individual)
        score += soft_score
        
        # 3. Actor efficiency (minimize idle days)
        actor_efficiency = self._calculate_actor_efficiency(individual)
        score += actor_efficiency * 100
        
        # 4. Location clustering bonus
        location_clustering = self._evaluate_location_clustering(individual)
        score += location_clustering * 200
        
        return score
    
    def _count_hard_violations(self, individual: np.ndarray) -> int:
        """Count hard constraint violations"""
        violations = 0
        
        for scene_idx, day_idx in enumerate(individual):
            scene = self.stripboard[scene_idx]
            
            # Check actor availability
            for actor in scene.get('Cast', []):
                if actor in self.actor_unavailable:
                    # Check if actor is unavailable on this day
                    # Simplified - need actual date checking
                    pass
            
            # Check location availability windows
            location = scene.get('Geographic_Location', '')
            if location in self.location_windows:
                window = self.location_windows[location]
                # Check if day is within window
                # Simplified - need actual date checking
                pass
        
        return violations
    
    def _evaluate_soft_constraints(self, individual: np.ndarray) -> float:
        """Evaluate soft constraint satisfaction"""
        score = 0.0
        # Evaluate each soft constraint
        return score
    
    def _calculate_actor_efficiency(self, individual: np.ndarray) -> float:
        """Calculate actor utilization efficiency"""
        actor_days = defaultdict(set)
        
        for scene_idx, day_idx in enumerate(individual):
            scene = self.stripboard[scene_idx]
            for actor in scene.get('Cast', []):
                actor_days[actor].add(day_idx)
        
        # Minimize gaps in actor schedules
        efficiency = 0.0
        for actor, days in actor_days.items():
            if len(days) > 1:
                days_list = sorted(list(days))
                gaps = sum(days_list[i+1] - days_list[i] - 1 for i in range(len(days_list)-1))
                efficiency -= gaps * 10
        
        return efficiency
    
    def _evaluate_location_clustering(self, individual: np.ndarray) -> float:
        """Reward clustering scenes at same location"""
        location_days = defaultdict(set)
        
        for scene_idx, day_idx in enumerate(individual):
            scene = self.stripboard[scene_idx]
            location = scene.get('Geographic_Location', '')
            location_days[location].add(day_idx)
        
        # Reward consecutive days at same location
        clustering_score = 0.0
        for location, days in location_days.items():
            if len(days) > 1:
                days_list = sorted(list(days))
                consecutive = sum(1 for i in range(len(days_list)-1) 
                                if days_list[i+1] - days_list[i] == 1)
                clustering_score += consecutive * 50
        
        return clustering_score
    
    def evolve(self) -> Tuple[np.ndarray, float]:
        """Run genetic algorithm for temporal scheduling"""
        pop_size = self.params.get('phase1_population', 50)
        generations = self.params.get('phase1_generations', 200)
        
        # Initialize population
        population = [self.create_individual() for _ in range(pop_size)]
        best_individual = None
        best_fitness = -float('inf')
        
        for gen in range(generations):
            # Evaluate fitness
            fitnesses = [self.fitness(ind) for ind in population]
            
            # Track best
            gen_best_idx = np.argmax(fitnesses)
            if fitnesses[gen_best_idx] > best_fitness:
                best_fitness = fitnesses[gen_best_idx]
                best_individual = population[gen_best_idx].copy()
            
            # Selection and reproduction
            new_population = []
            
            # Elitism
            elite_count = max(2, pop_size // 10)
            elite_indices = np.argsort(fitnesses)[-elite_count:]
            for idx in elite_indices:
                new_population.append(population[idx].copy())
            
            # Generate rest
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
        
        return best_individual, best_fitness
    
    def _tournament_selection(self, population: List, fitnesses: List) -> np.ndarray:
        """Tournament selection"""
        tournament_size = 5
        indices = self.rng.choice(len(population), tournament_size, replace=False)
        tournament_fits = [fitnesses[i] for i in indices]
        winner_idx = indices[np.argmax(tournament_fits)]
        return population[winner_idx].copy()
    
    def _crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
        """Uniform crossover for temporal assignment"""
        if self.rng.random() > self.params.get('crossover_rate', 0.85):
            return parent1.copy()
        
        child = parent1.copy()
        mask = self.rng.random(len(child)) < 0.5
        child[mask] = parent2[mask]
        
        return child
    
    def _mutate(self, individual: np.ndarray) -> np.ndarray:
        """Mutation for temporal assignment"""
        if self.rng.random() < self.params.get('mutation_rate', 0.02):
            # Random reassignment of a scene to different day
            scene_idx = self.rng.randint(len(individual))
            
            # Don't mutate director mandates
            scene_num = self.stripboard[scene_idx]['Scene_Number']
            if scene_num not in self.director_mandates:
                individual[scene_idx] = self.rng.randint(len(self.calendar.shooting_days))
        
        return individual

class Phase2GA:
    """Phase 2: Spatial Optimization - Optimize location sequence for each day"""
    
    def __init__(self, daily_scenes: Dict[int, List[Dict]], params: Dict):
        self.daily_scenes = daily_scenes  # day_idx -> list of scenes
        self.params = params
        self.rng = np.random.RandomState(params.get('seed', 42))
        
        # Build location distance matrix
        self._build_distance_matrix()
    
    def _build_distance_matrix(self):
        """Build distance estimates between locations"""
        # Extract unique locations
        all_locations = set()
        for scenes in self.daily_scenes.values():
            for scene in scenes:
                all_locations.add(scene.get('Geographic_Location', 'Unknown'))
        
        self.locations = list(all_locations)
        n = len(self.locations)
        
        # Estimate distances based on address similarity
        self.distance_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    self.distance_matrix[i][j] = self._estimate_distance(
                        self.locations[i], self.locations[j]
                    )
    
    def _estimate_distance(self, loc1: str, loc2: str) -> float:
        """Estimate distance between two locations"""
        if loc1 == loc2:
            return 0.0
        
        # Extract city/state for comparison
        city_pattern = r',\s*([^,]+),\s*[A-Z]{2}'
        
        city1 = re.search(city_pattern, loc1)
        city2 = re.search(city_pattern, loc2)
        
        if city1 and city2:
            if city1.group(1) == city2.group(1):
                return 5.0  # Same city
        
        state1 = re.search(r',\s*([A-Z]{2})', loc1)
        state2 = re.search(r',\s*([A-Z]{2})', loc2)
        
        if state1 and state2:
            if state1.group(1) == state2.group(1):
                return 30.0  # Same state
        
        return 100.0  # Different states or unknown
    
    def optimize_all_days(self) -> Dict[int, List[Dict]]:
        """Optimize location sequence for each shooting day"""
        optimized_schedule = {}
        
        for day_idx, scenes in self.daily_scenes.items():
            if len(scenes) <= 1:
                optimized_schedule[day_idx] = scenes
            else:
                # Optimize this day's shooting order
                optimized_order = self._optimize_single_day(scenes)
                optimized_schedule[day_idx] = optimized_order
        
        return optimized_schedule
    
    def _optimize_single_day(self, scenes: List[Dict]) -> List[Dict]:
        """Optimize location sequence for a single day"""
        if len(scenes) <= 2:
            return scenes
        
        # Group scenes by location
        location_groups = defaultdict(list)
        for scene in scenes:
            loc = scene.get('Geographic_Location', 'Unknown')
            location_groups[loc].append(scene)
        
        if len(location_groups) == 1:
            # All scenes at same location - just sort by scene number
            return sorted(scenes, key=lambda x: x.get('Scene_Number', ''))
        
        # Run GA to find optimal location visiting order
        best_order = self._ga_optimize_locations(list(location_groups.keys()))
        
        # Build final scene order
        ordered_scenes = []
        for location in best_order:
            # Add all scenes at this location (sorted by scene number)
            location_scenes = sorted(location_groups[location], 
                                    key=lambda x: x.get('Scene_Number', ''))
            ordered_scenes.extend(location_scenes)
        
        return ordered_scenes
    
    def _ga_optimize_locations(self, locations: List[str]) -> List[str]:
        """GA to optimize location visiting order"""
        if len(locations) <= 2:
            return locations
        
        n = len(locations)
        pop_size = min(20, self.params.get('phase2_population', 100))
        generations = min(100, self.params.get('phase2_generations', 500))
        
        # Create initial population (permutations)
        population = [self.rng.permutation(n).tolist() for _ in range(pop_size)]
        
        for gen in range(generations):
            # Evaluate fitness (minimize travel distance)
            fitnesses = []
            for individual in population:
                distance = sum(
                    self._get_location_distance(locations[individual[i]], 
                                               locations[individual[i+1]])
                    for i in range(len(individual)-1)
                )
                fitnesses.append(-distance)  # Negative because we minimize
            
            # Create new population
            new_population = []
            
            # Keep best
            best_idx = np.argmax(fitnesses)
            new_population.append(population[best_idx])
            
            # Generate rest
            while len(new_population) < pop_size:
                # Tournament selection
                parent1_idx = self._tournament(fitnesses, 3)
                parent2_idx = self._tournament(fitnesses, 3)
                
                # Order crossover
                child = self._order_crossover(
                    population[parent1_idx], 
                    population[parent2_idx]
                )
                
                # Swap mutation
                if self.rng.random() < 0.05:
                    i, j = self.rng.choice(n, 2, replace=False)
                    child[i], child[j] = child[j], child[i]
                
                new_population.append(child)
            
            population = new_population[:pop_size]
        
        # Return best order as locations
        best_individual = population[np.argmax(fitnesses)]
        return [locations[i] for i in best_individual]
    
    def _get_location_distance(self, loc1: str, loc2: str) -> float:
        """Get distance between two locations"""
        if loc1 in self.locations and loc2 in self.locations:
            i = self.locations.index(loc1)
            j = self.locations.index(loc2)
            return self.distance_matrix[i][j]
        return 10.0  # Default distance
    
    def _tournament(self, fitnesses: List[float], size: int) -> int:
        """Tournament selection"""
        indices = self.rng.choice(len(fitnesses), size, replace=False)
        tournament_fits = [fitnesses[i] for i in indices]
        return indices[np.argmax(tournament_fits)]
    
    def _order_crossover(self, parent1: List[int], parent2: List[int]) -> List[int]:
        """Order crossover for permutation"""
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

class ScheduleOptimizer:
    """Main orchestrator for 3-phase optimization"""
    
    def __init__(self, request: ScheduleRequest):
        self.stripboard = request.stripboard
        self.constraints_raw = request.constraints
        self.params = request.ga_params
        
        # Initialize components
        self.parser = StructuredConstraintParser()
        self.constraints = self.parser.parse_all_constraints(self.constraints_raw)
        
        print(f"DEBUG: PARSED {len(self.constraints)} CONSTRAINTS")
        for c in self.constraints[:10]:  # Show first 10
            print(f"  - {c.source.name}: {c.description}")

        # Determine shooting period from constraints or use defaults
        self.calendar = self._determine_calendar()
        
        # Detect conflicts
        self.conflict_detector = ConflictDetector(self.constraints, self.stripboard)
        self.conflicts = self.conflict_detector.detect_all_conflicts()

        print(f"DEBUG: DETECTED {len(self.conflicts)} CONFLICTS")
        print(f"DEBUG: CALENDAR HAS {len(self.calendar.shooting_days)} SHOOTING DAYS")
    
    def _determine_calendar(self) -> ShootingCalendar:
        """Determine shooting calendar from production parameters"""
        # Try to extract from operational data
        operational_data = self.constraints_raw.get('operational_data', {})
        
        # Look for production rules with date information
        if 'production_rules' in operational_data:
            # Try to parse dates from production rules
            # For now, use default
            pass
        
        # Default: 45 shooting days starting from Sept 1, 2025
        return ShootingCalendar("2025-09-01", "2025-10-31")
    
    def optimize(self) -> Dict[str, Any]:
        """Run 3-phase optimization"""
        import time
        start_time = time.time()
        
        # Phase 1: Temporal Scheduling
        phase1 = Phase1GA(self.stripboard, self.constraints, self.calendar, self.params)
        temporal_assignment, phase1_fitness = phase1.evolve()
        
        # Convert to daily scene groups
        daily_scenes = self._group_scenes_by_day(temporal_assignment)
        
        # Phase 2: Spatial Optimization
        phase2 = Phase2GA(daily_scenes, self.params)
        optimized_daily_schedule = phase2.optimize_all_days()
        
        # Build final schedule
        final_schedule = self._build_final_schedule(optimized_daily_schedule)
        
        # Calculate metrics
        metrics = self._calculate_metrics(final_schedule)
        
        processing_time = time.time() - start_time
        
        return {
            'schedule': final_schedule,
            'conflicts': [
                {
                    'scene': c.scene_id,
                    'severity': c.severity,
                    'issue': f"{c.constraint_1} vs {c.constraint_2}",
                    'resolution': c.suggested_resolution
                }
                for c in self.conflicts
            ],
            'metrics': metrics,
            'fitness_score': phase1_fitness,
            'processing_time_seconds': processing_time
        }
    
    def _group_scenes_by_day(self, temporal_assignment: np.ndarray) -> Dict[int, List[Dict]]:
        """Group scenes by assigned shooting day"""
        daily_scenes = defaultdict(list)
        
        for scene_idx, day_idx in enumerate(temporal_assignment):
            daily_scenes[int(day_idx)].append(self.stripboard[scene_idx])
        
        return dict(daily_scenes)
    
    def _build_final_schedule(self, daily_schedule: Dict[int, List[Dict]]) -> List[Dict]:
        """Build final day-by-day schedule"""
        schedule = []
        
        for day_idx in sorted(daily_schedule.keys()):
            shooting_date = self.calendar.shooting_days[day_idx]
            scenes = daily_schedule[day_idx]
            
            # Extract unique locations for this day
            locations_visited = []
            current_location = None
            for scene in scenes:
                loc = scene.get('Geographic_Location', 'Unknown')
                if loc != current_location:
                    locations_visited.append(loc)
                    current_location = loc
            
            schedule.append({
                'day': day_idx + 1,
                'date': shooting_date.strftime("%Y-%m-%d"),
                'locations': locations_visited,
                'scenes': scenes,
                'scene_count': len(scenes),
                'location_moves': len(locations_visited) - 1
            })
        
        return schedule
    
    def _calculate_metrics(self, schedule: List[Dict]) -> Dict[str, Any]:
        """Calculate schedule metrics"""
        total_moves = sum(day['location_moves'] for day in schedule)
        total_scenes = sum(day['scene_count'] for day in schedule)
        
        # Calculate location efficiency
        location_changes = []
        for day in schedule:
            location_changes.append(len(day['locations']))
        
        avg_locations_per_day = np.mean(location_changes) if location_changes else 0
        
        # Calculate constraint satisfaction
        satisfied_constraints = len([c for c in self.constraints 
                                    if c.type == ConstraintType.SOFT])
        total_constraints = len(self.constraints)
        satisfaction_rate = satisfied_constraints / total_constraints if total_constraints > 0 else 0
        
        return {
            'total_shooting_days': len(schedule),
            'total_scenes': total_scenes,
            'total_location_moves': total_moves,
            'avg_locations_per_day': round(avg_locations_per_day, 2),
            'constraint_satisfaction_rate': round(satisfaction_rate, 2),
            'hard_conflicts': len([c for c in self.conflicts if c.severity == "HIGH"]),
            'soft_conflicts': len([c for c in self.conflicts if c.severity != "HIGH"])
        }

@app.post("/optimize/schedule", response_model=ScheduleResponse)
async def optimize_schedule(request: ScheduleRequest):
    """
    Advanced 3-phase film schedule optimization
    Updated to work with structured constraints from n8n
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
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "version": "3.0.0", "phases": 3, "parser": "structured"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
