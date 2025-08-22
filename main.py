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
from datetime import datetime, timedelta, date, time
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
    """Response model with optimized schedule - UPDATED with conflicts_summary"""
    schedule: List[Dict[str, Any]]
    summary: Dict[str, Any]
    missing_scenes: Dict[str, Any]
    conflicts: List[Dict[str, Any]]
    conflicts_summary: Dict[str, Any]        # NEW: Add this field
    metrics: Dict[str, Any]
    fitness_score: float
    processing_time_seconds: float

class StructuredConstraintParser:
    """Parses structured constraints from n8n AI agents"""
    
    def __init__(self):
        # Existing parsing diagnostics
        self.parsing_stats = {
            'structured_v2_count': 0,
            'legacy_v1_count': 0, 
            'legacy_list_count': 0,
            'fallback_attempts': 0,
            'fallback_successes': 0,
            'fallback_failures': 0,
            'constraint_type_detection': {
                'shoot_first': 0,
                'shoot_last': 0,
                'sequence': 0,
                'same_day': 0,
                'location_grouping': 0,
                'unrecognized': 0
            },
            'failed_constraints': []
        }
        
        # Initialize constraints list
        self.constraints = []
        
        # NEW: Location constraint parsing statistics
        self.location_parsing_stats = {
            'total_location_constraints': 0,
            'availability_windows_parsed': 0,
            'access_restrictions_parsed': 0,
            'environmental_factors_parsed': 0,
            'parsing_failures': 0,
            'constraint_categories': {
                'Availability': 0,
                'Sound': 0,
                'Power': 0,
                'Parking': 0,
                'Access': 0,
                'Lighting': 0,
                'General Notes': 0,
                'Other': 0
            }
        }
    
    def parse_all_constraints(self, constraints_dict: Dict[str, Any]) -> List[Constraint]:
        """Parse all structured constraint groups from n8n - ENHANCED Phase A logging"""
        all_constraints = []
        
        try:
            if 'people_constraints' in constraints_dict:
                people_constraints = self._parse_people_constraints(constraints_dict['people_constraints'])
                all_constraints.extend(people_constraints)
            
            if 'location_constraints' in constraints_dict:
                location_constraints = self._parse_location_constraints(constraints_dict['location_constraints'])
                all_constraints.extend(location_constraints)
            
            if 'technical_constraints' in constraints_dict:
                technical_constraints = self._parse_technical_constraints(constraints_dict['technical_constraints'])
                all_constraints.extend(technical_constraints)
            
            if 'creative_constraints' in constraints_dict:
                creative_constraints = self._parse_creative_constraints(constraints_dict['creative_constraints'])
                all_constraints.extend(creative_constraints)
            
            if 'operational_data' in constraints_dict:
                operational_constraints = self._parse_operational_data(constraints_dict['operational_data'])
                all_constraints.extend(operational_constraints)
        
        except Exception as e:
            print(f"ERROR: Constraint parsing failed: {e}")
        
        # Enhanced summary with source breakdown
        source_counts = {}
        for constraint in all_constraints:
            source = constraint.source.name
            source_counts[source] = source_counts.get(source, 0) + 1
        
        print(f"DEBUG: Parsed {len(all_constraints)} total constraints by source: {source_counts}")
        return all_constraints
    
    def _parse_people_constraints(self, people_data: Dict) -> List[Constraint]:
        """Parse actor availability constraints - NO VERBOSE LOGGING"""
        constraints = []
        
        try:
            if 'actors' in people_data:
                actors_data = people_data['actors']
                
                if isinstance(actors_data, dict):
                    for actor_name, actor_info in actors_data.items():
                        if not isinstance(actor_info, dict):
                            continue
                        
                        constraint_level = actor_info.get('constraint_level', 'Hard')
                        constraint_type = ConstraintType.HARD if constraint_level == 'Hard' else ConstraintType.SOFT
                        
                        # Parse unavailable dates (NO DEBUG PRINTS)
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
                        
                        # Parse week restrictions (NO DEBUG PRINTS)
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
                        
                        # Parse daily restrictions (NO DEBUG PRINTS)
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
            
            # Store cast_mapping (NO DEBUG PRINTS)
            if 'cast_mapping' in people_data:
                cast_mapping_data = people_data['cast_mapping']
                
                constraints.append(Constraint(
                    source=ConstraintPriority.ACTOR,
                    type=ConstraintType.SOFT,
                    description="Cast mapping data",
                    affected_scenes=[],
                    actor_restriction={
                        'cast_mapping': cast_mapping_data
                    }
                ))
        
        except Exception as e:
            print(f"ERROR: People constraints parsing failed: {e}")
        
        return constraints
    
    def _parse_location_constraints(self, location_data: Dict) -> List[Constraint]:
        """Parse location availability and travel time constraints - ENHANCED Phase A"""
        constraints = []
        
        try:
            # Parse location availability with enhanced structured parsing
            if 'locations' in location_data:
                locations_info = location_data['locations']
                
                if isinstance(locations_info, dict):
                    for location_name, location_info in locations_info.items():
                        try:
                            if isinstance(location_info, dict) and 'constraints' in location_info:
                                location_constraints = location_info['constraints']
                                
                                if isinstance(location_constraints, list):
                                    for constraint_info in location_constraints:
                                        try:
                                            if isinstance(constraint_info, dict):
                                                # Track parsing attempt
                                                self.location_parsing_stats['total_location_constraints'] += 1
                                                
                                                # Extract basic constraint info
                                                raw_category = constraint_info.get('category', 'Other')
                                                details = constraint_info.get('details', '')
                                                constraint_level = constraint_info.get('constraint_level', 'Hard')
                                                
                                                # Categorize constraint type
                                                structured_category = self._categorize_location_constraint(raw_category, details)
                                                self.location_parsing_stats['constraint_categories'][structured_category] += 1
                                                
                                                # Parse structured data based on category
                                                parsed_data = self._parse_location_constraint_details(structured_category, details)
                                                
                                                # Track parsing success
                                                if parsed_data.get('parsed_successfully'):
                                                    if structured_category == 'Availability':
                                                        self.location_parsing_stats['availability_windows_parsed'] += 1
                                                    elif structured_category in ['Sound', 'Power', 'Parking', 'Access']:
                                                        self.location_parsing_stats['access_restrictions_parsed'] += 1
                                                    elif structured_category in ['Lighting', 'General Notes']:
                                                        self.location_parsing_stats['environmental_factors_parsed'] += 1
                                                else:
                                                    self.location_parsing_stats['parsing_failures'] += 1
                                                
                                                # Create enhanced constraint object
                                                constraint_type = ConstraintType.HARD if constraint_level == 'Hard' else ConstraintType.SOFT
                                                
                                                constraint = self._create_safe_constraint(
                                                    source=ConstraintPriority.LOCATION,
                                                    constraint_type=constraint_type,
                                                    description=f"{location_name}: {details}",
                                                    affected_scenes=[],
                                                    location_restriction={
                                                        'location': location_name,
                                                        'category': structured_category,
                                                        'raw_category': raw_category,
                                                        'details': details,
                                                        'parsed_data': parsed_data  # NEW: Structured parsed data
                                                    }
                                                )
                                                
                                                if constraint:
                                                    constraints.append(constraint)
                                                    print(f"DEBUG: Parsed {structured_category} constraint for {location_name}: {parsed_data.get('summary', 'No summary')}")
                                        
                                        except Exception as e:
                                            print(f"ERROR: Failed to parse location constraint for {location_name}: {e}")
                                            self.location_parsing_stats['parsing_failures'] += 1
                                            continue
                        
                        except Exception as e:
                            print(f"ERROR: Failed to process location {location_name}: {e}")
                            continue
            
            # Parse travel times (existing logic with error handling)
            if 'travel_times' in location_data:
                travel_data = location_data['travel_times']
                
                if isinstance(travel_data, list):
                    for travel_info in travel_data:
                        try:
                            if isinstance(travel_info, dict):
                                constraint = self._create_safe_constraint(
                                    source=ConstraintPriority.LOCATION,
                                    constraint_type=ConstraintType.SOFT,
                                    description=f"Travel time: {travel_info.get('estimated_travel_time_minutes', 0)} minutes",
                                    affected_scenes=[],
                                    location_restriction={
                                        'from_location': travel_info.get('from_location_fictional', ''),
                                        'to_location': travel_info.get('to_location_fictional', ''),
                                        'travel_time_minutes': travel_info.get('estimated_travel_time_minutes', 0),
                                        'category': 'Travel',
                                        'parsed_data': {
                                            'constraint_type': 'travel_time',
                                            'travel_minutes': travel_info.get('estimated_travel_time_minutes', 0),
                                            'parsed_successfully': True
                                        }
                                    }
                                )
                                
                                if constraint:
                                    constraints.append(constraint)
                        
                        except Exception as e:
                            print(f"ERROR: Failed to parse travel time constraint: {e}")
                            continue
        
        except Exception as e:
            print(f"ERROR: Location constraints parsing failed: {e}")
        
        # Print parsing statistics
        self._print_location_parsing_stats()
        
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
    
    # PHASE A: DATA STRUCTURE HANDLING - New Director Notes JSON Structure
    # Replace this method in StructuredConstraintParser class

    def _parse_creative_constraints(self, creative_data: Dict) -> List[Constraint]:
        """Parse director and DOP constraints - ENHANCED with comprehensive error handling"""
        constraints = []
        
        try:
            # Parse director constraints with comprehensive error handling
            if 'director_notes' in creative_data:
                director_data = creative_data['director_notes']
                print(f"DEBUG: Director data type: {type(director_data)}")
                
                # Handle new structured director constraints from two-agent system
                if isinstance(director_data, dict) and 'director_constraints' in director_data:
                    director_constraints = director_data['director_constraints']
                    print(f"DEBUG: Found structured director_constraints array with {len(director_constraints)} items")
                    
                    if isinstance(director_constraints, list):
                        for i, constraint_info in enumerate(director_constraints):
                            try:
                                if not isinstance(constraint_info, dict):
                                    print(f"WARNING: Skipping non-dict constraint {i}: {type(constraint_info)}")
                                    continue
                                
                                # Track structured parsing
                                self.parsing_stats['structured_v2_count'] += 1
                                
                                # Validate and extract structured data with error handling
                                constraint_type = constraint_info.get('constraint_type', '')
                                if not constraint_type:
                                    print(f"WARNING: Skipping constraint {i} - missing constraint_type")
                                    continue
                                
                                constraint_level = constraint_info.get('constraint_level', 'Hard')
                                constraint_text = constraint_info.get('constraint_text', '')
                                related_scenes = constraint_info.get('related_scenes', [])
                                related_locations = constraint_info.get('related_locations', [])
                                reasoning = constraint_info.get('reasoning', '')
                                
                                # Validate and clean data types
                                safe_scenes = self._validate_scene_list(related_scenes, f"constraint {i}")
                                safe_locations = self._validate_location_list(related_locations, f"constraint {i}")
                                constraint_type_enum = self._validate_constraint_level(constraint_level)
                                
                                print(f"DEBUG: STRUCTURED constraint {i+1}: type='{constraint_type}', "
                                    f"scenes={safe_scenes}, level='{constraint_level}'")
                                
                                # Create constraint safely
                                constraint = self._create_safe_constraint(
                                    source=ConstraintPriority.DIRECTOR,
                                    constraint_type=constraint_type_enum,
                                    description=constraint_text,
                                    affected_scenes=safe_scenes,
                                    date_restriction={
                                        'constraint_type': constraint_type,
                                        'locations': safe_locations,
                                        'reasoning': reasoning,
                                        'source_format': 'structured_v2'
                                    }
                                )
                                
                                if constraint:
                                    constraints.append(constraint)
                            
                            except Exception as e:
                                print(f"ERROR: Failed to process structured constraint {i}: {e}")
                                continue  # Skip this constraint, continue with others
                    
                    # FALLBACK: Handle old formats with enhanced error handling
                    elif isinstance(director_data, dict) and 'director_constraints' in director_data:
                        try:
                            self.parsing_stats['legacy_v1_count'] += 1
                            print(f"DEBUG: Using LEGACY V1 format parsing")
                            
                            old_director_constraints = director_data['director_constraints']
                            if isinstance(old_director_constraints, list):
                                for j, constraint_info in enumerate(old_director_constraints):
                                    try:
                                        if isinstance(constraint_info, dict):
                                            constraint_level = constraint_info.get('constraint_level', 'Hard')
                                            constraint_type_enum = self._validate_constraint_level(constraint_level)
                                            safe_scenes = self._validate_scene_list(
                                                constraint_info.get('related_scenes', []), f"legacy constraint {j}")
                                            safe_locations = self._validate_location_list(
                                                constraint_info.get('related_locations', []), f"legacy constraint {j}")
                                            
                                            constraint = self._create_safe_constraint(
                                                source=ConstraintPriority.DIRECTOR,
                                                constraint_type=constraint_type_enum,
                                                description=constraint_info.get('constraint_text', ''),
                                                affected_scenes=safe_scenes,
                                                date_restriction={
                                                    'category': constraint_info.get('category'),
                                                    'locations': safe_locations,
                                                    'source_format': 'legacy_v1'
                                                }
                                            )
                                            
                                            if constraint:
                                                constraints.append(constraint)
                                    except Exception as e:
                                        print(f"ERROR: Failed to process legacy v1 constraint {j}: {e}")
                                        continue
                        except Exception as e:
                            print(f"ERROR: Legacy v1 processing failed: {e}")
                    
                    elif isinstance(director_data, list):
                        try:
                            self.parsing_stats['legacy_list_count'] += 1
                            print(f"DEBUG: Using LEGACY LIST format parsing")
                            
                            for k, constraint_info in enumerate(director_data):
                                try:
                                    if isinstance(constraint_info, dict):
                                        constraint_level = constraint_info.get('constraint_level', 'Hard')
                                        constraint_type_enum = self._validate_constraint_level(constraint_level)
                                        safe_scenes = self._validate_scene_list(
                                            constraint_info.get('related_scenes', []), f"legacy list constraint {k}")
                                        safe_locations = self._validate_location_list(
                                            constraint_info.get('related_locations', []), f"legacy list constraint {k}")
                                        
                                        constraint = self._create_safe_constraint(
                                            source=ConstraintPriority.DIRECTOR,
                                            constraint_type=constraint_type_enum,
                                            description=constraint_info.get('constraint_text', ''),
                                            affected_scenes=safe_scenes,
                                            date_restriction={
                                                'category': constraint_info.get('category'),
                                                'locations': safe_locations,
                                                'source_format': 'legacy_list'
                                            }
                                        )
                                        
                                        if constraint:
                                            constraints.append(constraint)
                                except Exception as e:
                                    print(f"ERROR: Failed to process legacy list constraint {k}: {e}")
                                    continue
                        except Exception as e:
                            print(f"ERROR: Legacy list processing failed: {e}")
                    
                    else:
                        print(f"DEBUG: UNRECOGNIZED director_data format: {type(director_data)}")
                        print(f"DEBUG: Director data keys: {list(director_data.keys()) if isinstance(director_data, dict) else 'Not a dict'}")
                
                else:
                    print(f"DEBUG: No 'director_notes' found in creative_data")
            
            # Parse DOP constraints with error handling
            if 'dop_priorities' in creative_data:
                try:
                    dop_data = creative_data['dop_priorities']
                    
                    if isinstance(dop_data, dict) and 'dop_priorities' in dop_data:
                        dop_constraints = dop_data['dop_priorities']
                    elif isinstance(dop_data, list):
                        dop_constraints = dop_data
                    else:
                        dop_constraints = []
                    
                    if isinstance(dop_constraints, list):
                        for m, constraint_info in enumerate(dop_constraints):
                            try:
                                if isinstance(constraint_info, dict):
                                    constraint_level = constraint_info.get('constraint_level', 'Hard')
                                    constraint_type = self._validate_constraint_level(constraint_level)
                                    safe_scenes = self._validate_scene_list(
                                        constraint_info.get('related_scenes', []), f"DOP constraint {m}")
                                    safe_locations = self._validate_location_list(
                                        constraint_info.get('related_locations', []), f"DOP constraint {m}")
                                    
                                    constraint = self._create_safe_constraint(
                                        source=ConstraintPriority.DOP,
                                        constraint_type=constraint_type,
                                        description=constraint_info.get('constraint_text', ''),
                                        affected_scenes=safe_scenes,
                                        location_restriction={
                                            'category': constraint_info.get('category'),
                                            'locations': safe_locations
                                        }
                                    )
                                    
                                    if constraint:
                                        constraints.append(constraint)
                            except Exception as e:
                                print(f"ERROR: Failed to process DOP constraint {m}: {e}")
                                continue
                except Exception as e:
                    print(f"ERROR: DOP constraints processing failed: {e}")
        
        except Exception as e:
            print(f"ERROR: Creative constraints parsing failed: {e}")
            import traceback
            traceback.print_exc()
        
        # Enhanced summary with parsing method breakdown
        director_count = len([c for c in constraints if c.source == ConstraintPriority.DIRECTOR])
        dop_count = len([c for c in constraints if c.source == ConstraintPriority.DOP])
        
        print(f"DEBUG: Creative constraints parsed - {director_count} director, {dop_count} DOP constraints")
        print(f"DEBUG: Parsing method stats - Structured_v2: {self.parsing_stats['structured_v2_count']}, "
            f"Legacy_v1: {self.parsing_stats['legacy_v1_count']}, Legacy_list: {self.parsing_stats['legacy_list_count']}")
        
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

    def _validate_scene_list(self, scenes: Any, context: str) -> List[str]:
        """Safely convert and validate scene number list"""
        if scenes is None:
            return []
        
        if not isinstance(scenes, list):
            print(f"WARNING: {context} - converting {type(scenes)} to list")
            scenes = [scenes] if scenes else []
        
        validated_scenes = []
        for scene in scenes:
            try:
                if scene is not None:
                    scene_str = str(scene).strip()
                    if scene_str:
                        validated_scenes.append(scene_str)
            except Exception as e:
                print(f"WARNING: {context} - invalid scene number '{scene}': {e}")
        
        return validated_scenes

    def _validate_location_list(self, locations: Any, context: str) -> List[str]:
        """Safely convert and validate location list"""
        if locations is None:
            return []
        
        if not isinstance(locations, list):
            print(f"WARNING: {context} - converting locations {type(locations)} to list")
            locations = [locations] if locations else []
        
        validated_locations = []
        for location in locations:
            try:
                if location is not None:
                    location_str = str(location).strip()
                    if location_str:
                        validated_locations.append(location_str)
            except Exception as e:
                print(f"WARNING: {context} - invalid location '{location}': {e}")
        
        return validated_locations

    def _validate_constraint_level(self, level: Any) -> ConstraintType:
        """Safely validate and convert constraint level"""
        if level is None:
            return ConstraintType.HARD
        
        if isinstance(level, ConstraintType):
            return level
        
        try:
            level_str = str(level).lower().strip()
            if level_str in ['hard', 'h']:
                return ConstraintType.HARD
            elif level_str in ['soft', 's']:
                return ConstraintType.SOFT
            else:
                print(f"WARNING: Unknown constraint level '{level}', defaulting to Hard")
                return ConstraintType.HARD
        except Exception as e:
            print(f"WARNING: Error validating constraint level '{level}': {e}, defaulting to Hard")
            return ConstraintType.HARD

    def _create_safe_constraint(self, source: ConstraintPriority, constraint_type: ConstraintType, 
                          description: str, affected_scenes: List[str], 
                          date_restriction: Optional[Dict] = None,
                          location_restriction: Optional[Dict] = None,
                          actor_restriction: Optional[Dict] = None) -> Optional[Constraint]:
        """Create constraint with validation - returns None if validation fails"""
        try:
            # Validate inputs
            if not isinstance(description, str):
                print(f"WARNING: Invalid description type: {type(description)}")
                description = str(description) if description else "Unknown constraint"
            
            if not isinstance(affected_scenes, list):
                print(f"WARNING: Converting affected_scenes to list: {type(affected_scenes)}")
                affected_scenes = [affected_scenes] if affected_scenes else []
            
            # Ensure all scene numbers are strings and valid
            safe_scenes = []
            for scene in affected_scenes:
                try:
                    if scene is not None and str(scene).strip():
                        safe_scenes.append(str(scene).strip())
                except Exception as e:
                    print(f"WARNING: Invalid scene in affected_scenes '{scene}': {e}")
            
            return Constraint(
                source=source,
                type=constraint_type,
                description=description,
                affected_scenes=safe_scenes,
                date_restriction=date_restriction,
                location_restriction=location_restriction,
                actor_restriction=actor_restriction
            )
        
        except Exception as e:
            print(f"ERROR: Failed to create constraint: {e}")
            return None

    def _parse_date_range(self, details_text: str) -> Dict[str, Any]:
        """Extract date ranges from location constraint details"""
        date_info = {
            'has_date_restriction': False,
            'start_date': None,
            'end_date': None,
            'restricted_dates': [],
            'available_dates': []
        }
        
        try:
            # Pattern 1: "Available Sept 22-24" or "Available from Sept 22 to Sept 24"
            date_range_patterns = [
                r'available\s+(?:from\s+)?(\w+\s+\d+)(?:\s*-\s*|\s+to\s+)(\w+\s+\d+)',
                r'available\s+(\w+\s+\d+)\s*-\s*(\w+\s+\d+)',
                r'(\w+\s+\d+)\s*-\s*(\w+\s+\d+)\s+only',
                r'from\s+(\w+\s+\d+)\s+to\s+(\w+\s+\d+)'
            ]
            
            for pattern in date_range_patterns:
                match = re.search(pattern, details_text.lower())
                if match:
                    start_str, end_str = match.groups()
                    start_date = self._parse_date_string(start_str)
                    end_date = self._parse_date_string(end_str)
                    
                    if start_date and end_date:
                        date_info['has_date_restriction'] = True
                        date_info['start_date'] = start_date.strftime("%Y-%m-%d")
                        date_info['end_date'] = end_date.strftime("%Y-%m-%d")
                        break
            
            # Pattern 2: "No access on Sept 25" or "Unavailable Sept 26-27"
            restriction_patterns = [
                r'no\s+(?:access|filming)\s+on\s+(\w+\s+\d+)',
                r'unavailable\s+(\w+\s+\d+)(?:\s*-\s*(\w+\s+\d+))?',
                r'closed\s+(\w+\s+\d+)'
            ]
            
            for pattern in restriction_patterns:
                matches = re.finditer(pattern, details_text.lower())
                for match in matches:
                    if match.group(2):  # Date range
                        start_date = self._parse_date_string(match.group(1))
                        end_date = self._parse_date_string(match.group(2))
                        if start_date and end_date:
                            current = start_date
                            while current <= end_date:
                                date_info['restricted_dates'].append(current.strftime("%Y-%m-%d"))
                                current += timedelta(days=1)
                    else:  # Single date
                        restricted_date = self._parse_date_string(match.group(1))
                        if restricted_date:
                            date_info['restricted_dates'].append(restricted_date.strftime("%Y-%m-%d"))
                    
                    date_info['has_date_restriction'] = True
            
            # Pattern 3: "Available [specific dates]"
            specific_date_patterns = [
                r'available\s+(\w+\s+\d+(?:,\s*\w+\s+\d+)*)',
                r'only\s+(\w+\s+\d+(?:,\s*\w+\s+\d+)*)\s+available'
            ]
            
            for pattern in specific_date_patterns:
                match = re.search(pattern, details_text.lower())
                if match:
                    date_list = match.group(1)
                    for date_str in re.split(r',\s*', date_list):
                        available_date = self._parse_date_string(date_str.strip())
                        if available_date:
                            date_info['available_dates'].append(available_date.strftime("%Y-%m-%d"))
                            date_info['has_date_restriction'] = True
        
        except Exception as e:
            print(f"WARNING: Date parsing failed for '{details_text}': {e}")
        
        return date_info

    def _parse_time_restrictions(self, details_text: str) -> Dict[str, Any]:
        """Extract time restrictions from location constraint details"""
        time_info = {
            'has_time_restriction': False,
            'start_time': None,
            'end_time': None,
            'daily_hours': None,
            'restricted_times': []
        }
        
        try:
            # Pattern 1: "07:00 to 20:00" or "7am-8pm" 
            time_range_patterns = [
                r'(\d{1,2}:\d{2})\s*(?:to|-)?\s*(\d{1,2}:\d{2})',
                r'(\d{1,2}(?:am|pm))\s*(?:to|-)\s*(\d{1,2}(?:am|pm))',
                r'from\s+(\d{1,2}:\d{2})\s+to\s+(\d{1,2}:\d{2})',
                r'between\s+(\d{1,2}(?:am|pm))\s+and\s+(\d{1,2}(?:am|pm))'
            ]
            
            for pattern in time_range_patterns:
                match = re.search(pattern, details_text.lower())
                if match:
                    start_time_str, end_time_str = match.groups()
                    start_time = self._parse_time_string(start_time_str)
                    end_time = self._parse_time_string(end_time_str)
                    
                    if start_time and end_time:
                        time_info['has_time_restriction'] = True
                        time_info['start_time'] = start_time.strftime("%H:%M")
                        time_info['end_time'] = end_time.strftime("%H:%M")
                        
                        # Calculate daily hours
                        start_datetime = datetime.combine(date.today(), start_time)
                        end_datetime = datetime.combine(date.today(), end_time)
                        if end_datetime < start_datetime:  # Next day
                            end_datetime += timedelta(days=1)
                        duration = end_datetime - start_datetime
                        time_info['daily_hours'] = duration.total_seconds() / 3600
                        break
            
            # Pattern 2: "No filming after 10pm" or "Noisy in the mornings"
            restriction_patterns = [
                r'no\s+filming\s+(?:after|past)\s+(\d{1,2}(?::\d{2})?(?:am|pm)?)',
                r'no\s+access\s+(?:before|until)\s+(\d{1,2}(?::\d{2})?(?:am|pm)?)',
                r'noisy\s+(?:in\s+the\s+)?(morning|afternoon|evening)s?',
                r'busy\s+(?:from\s+)?(\d{1,2}(?::\d{2})?(?:am|pm)?)\s*-\s*(\d{1,2}(?::\d{2})?(?:am|pm)?)'
            ]
            
            for pattern in restriction_patterns:
                match = re.search(pattern, details_text.lower())
                if match:
                    time_info['has_time_restriction'] = True
                    if 'morning' in match.group(0):
                        time_info['restricted_times'].append({'period': 'morning', 'reason': 'noisy'})
                    elif 'afternoon' in match.group(0):
                        time_info['restricted_times'].append({'period': 'afternoon', 'reason': 'noisy'})
                    elif 'evening' in match.group(0):
                        time_info['restricted_times'].append({'period': 'evening', 'reason': 'noisy'})
                    elif match.groups():
                        time_info['restricted_times'].append({
                            'time': match.group(1),
                            'type': 'after' if 'after' in match.group(0) else 'before'
                        })
        
        except Exception as e:
            print(f"WARNING: Time parsing failed for '{details_text}': {e}")
        
        return time_info

    def _parse_access_restrictions(self, details_text: str) -> Dict[str, Any]:
        """Extract access and logistical restrictions"""
        access_info = {
            'has_access_restrictions': False,
            'power_limitations': [],
            'parking_restrictions': [],
            'sound_issues': [],
            'equipment_limitations': [],
            'crew_size_limits': [],
            'special_requirements': []
        }
        
        try:
            # Power restrictions
            power_patterns = [
                r'limited\s+(?:power|outlets|electricity)',
                r'no\s+(?:power|electricity|outlets)',
                r'generator\s+required',
                r'power\s+(?:issues|problems|limitations)'
            ]
            
            for pattern in power_patterns:
                if re.search(pattern, details_text.lower()):
                    access_info['has_access_restrictions'] = True
                    access_info['power_limitations'].append(pattern.replace('\\s+', ' '))
            
            # Parking restrictions  
            parking_patterns = [
                r'(?:no|limited)\s+(?:crew\s+)?parking',
                r'parking\s+(?:issues|restrictions|problems)',
                r'no\s+(?:truck|vehicle)\s+access',
                r'street\s+parking\s+only'
            ]
            
            for pattern in parking_patterns:
                if re.search(pattern, details_text.lower()):
                    access_info['has_access_restrictions'] = True
                    access_info['parking_restrictions'].append(pattern.replace('\\s+', ' '))
            
            # Sound issues
            sound_patterns = [
                r'(?:high\s+traffic|busy\s+road|noisy)',
                r'sound\s+(?:issues|problems)',
                r'(?:airplane|plane|aircraft)\s+noise',
                r'construction\s+nearby'
            ]
            
            for pattern in sound_patterns:
                if re.search(pattern, details_text.lower()):
                    access_info['has_access_restrictions'] = True
                    access_info['sound_issues'].append(pattern.replace('\\s+', ' '))
            
            # Equipment limitations
            equipment_patterns = [
                r'no\s+(?:crane|dolly|steadicam)\s+access',
                r'(?:stairs|narrow)\s+access\s+only',
                r'equipment\s+(?:limitations|restrictions)',
                r'small\s+crew\s+only'
            ]
            
            for pattern in equipment_patterns:
                if re.search(pattern, details_text.lower()):
                    access_info['has_access_restrictions'] = True
                    access_info['equipment_limitations'].append(pattern.replace('\\s+', ' '))
            
            # Crew size limits
            crew_patterns = [
                r'(?:small|minimal)\s+crew\s+only',
                r'maximum\s+(\d+)\s+people',
                r'crew\s+size\s+(?:limited|restricted)',
                r'intimate\s+setting'
            ]
            
            for pattern in crew_patterns:
                match = re.search(pattern, details_text.lower())
                if match:
                    access_info['has_access_restrictions'] = True
                    if match.groups():
                        access_info['crew_size_limits'].append({'max_people': int(match.group(1))})
                    else:
                        access_info['crew_size_limits'].append({'restriction': pattern.replace('\\s+', ' ')})
            
            # Special requirements
            special_patterns = [
                r'(?:permit|permission)\s+required',
                r'insurance\s+(?:required|needed)',
                r'security\s+(?:required|needed)',
                r'advance\s+(?:notice|booking)',
                r'escort\s+required'
            ]
            
            for pattern in special_patterns:
                if re.search(pattern, details_text.lower()):
                    access_info['has_access_restrictions'] = True
                    access_info['special_requirements'].append(pattern.replace('\\s+', ' '))
        
        except Exception as e:
            print(f"WARNING: Access restriction parsing failed for '{details_text}': {e}")
        
        return access_info


    def _categorize_location_constraint(self, category: str, details: str) -> str:
        """Categorize location constraint into structured type"""
        
        # Clean and normalize category
        if not category:
            category = "Other"
        
        category_lower = category.lower().strip()
        details_lower = details.lower() if details else ""
        
        # Map categories to standardized types
        if category_lower in ['availability', 'available', 'dates']:
            return 'Availability'
        elif category_lower in ['sound', 'noise', 'audio']:
            return 'Sound'  
        elif category_lower in ['power', 'electricity', 'electrical']:
            return 'Power'
        elif category_lower in ['parking', 'vehicles', 'access']:
            return 'Parking'
        elif category_lower in ['lighting', 'light', 'natural light']:
            return 'Lighting'
        elif category_lower in ['general notes', 'notes', 'general']:
            return 'General Notes'
        
        # Content-based categorization if category is unclear
        if any(word in details_lower for word in ['available', 'date', 'time', 'hour']):
            return 'Availability'
        elif any(word in details_lower for word in ['noisy', 'sound', 'traffic', 'quiet']):
            return 'Sound'
        elif any(word in details_lower for word in ['power', 'outlet', 'electricity', 'generator']):
            return 'Power'
        elif any(word in details_lower for word in ['parking', 'vehicle', 'truck', 'car']):
            return 'Parking'
        elif any(word in details_lower for word in ['access', 'entrance', 'stairs', 'narrow']):
            return 'Access'
        elif any(word in details_lower for word in ['light', 'lighting', 'morning', 'golden hour']):
            return 'Lighting'
        else:
            return 'Other'


    def _parse_date_string(self, date_str: str) -> Optional[date]:
        """Parse various date string formats"""
        if not date_str:
            return None
        
        try:
            # Clean the string
            date_str = date_str.strip()
            
            # Common patterns with current year assumption
            current_year = datetime.now().year
            
            patterns = [
                ("%B %d", f"%B %d {current_year}"),      # "September 22" -> "September 22 2025"
                ("%b %d", f"%b %d {current_year}"),      # "Sept 22" -> "Sept 22 2025"  
                ("%m/%d", f"%m/%d/{current_year}"),      # "9/22" -> "9/22/2025"
                ("%m-%d", f"%m-%d-{current_year}"),      # "9-22" -> "9-22-2025"
                ("%Y-%m-%d", "%Y-%m-%d"),                # "2025-09-22"
                ("%m/%d/%Y", "%m/%d/%Y"),                # "9/22/2025"
                ("%B %d, %Y", "%B %d, %Y"),              # "September 22, 2025"
            ]
            
            for input_pattern, full_pattern in patterns:
                try:
                    # Try to parse with full format first
                    if input_pattern == full_pattern:
                        return datetime.strptime(date_str, input_pattern).date()
                    else:
                        # Add current year and parse
                        full_date_str = f"{date_str} {current_year}" if input_pattern.endswith(" %d") else date_str
                        return datetime.strptime(full_date_str, full_pattern).date()
                except ValueError:
                    continue
            
            print(f"WARNING: Could not parse date string: '{date_str}'")
            return None
        
        except Exception as e:
            print(f"WARNING: Date parsing error for '{date_str}': {e}")
            return None

    def _parse_time_string(self, time_str: str) -> Optional[time]:
        """Parse various time string formats"""
        if not time_str:
            return None
        
        try:
            time_str = time_str.strip()
            
            patterns = [
                "%H:%M",        # "14:30"
                "%I:%M%p",      # "2:30PM"
                "%I%p",         # "2PM"
                "%H",           # "14"
            ]
            
            for pattern in patterns:
                try:
                    return datetime.strptime(time_str.upper(), pattern).time()
                except ValueError:
                    continue
            
            print(f"WARNING: Could not parse time string: '{time_str}'")
            return None
    
        except Exception as e:
            print(f"WARNING: Time parsing error for '{time_str}': {e}")
            return None

    def _parse_location_constraint_details(self, category: str, details: str) -> Dict[str, Any]:
        """Parse location constraint details into structured data"""
        parsed_data = {
            'constraint_type': category.lower(),
            'parsed_successfully': False,
            'summary': ''
        }
        
        try:
            if category == 'Availability':
                # Parse availability windows
                date_info = self._parse_date_range(details)
                time_info = self._parse_time_restrictions(details)
                
                parsed_data.update({
                    'date_restrictions': date_info,
                    'time_restrictions': time_info,
                    'parsed_successfully': date_info['has_date_restriction'] or time_info['has_time_restriction'],
                    'summary': self._create_availability_summary(date_info, time_info)
                })
            
            elif category in ['Sound', 'Power', 'Parking', 'Access']:
                # Parse access restrictions
                access_info = self._parse_access_restrictions(details)
                time_info = self._parse_time_restrictions(details)  # Time-based restrictions
                
                parsed_data.update({
                    'access_restrictions': access_info,
                    'time_restrictions': time_info,
                    'parsed_successfully': access_info['has_access_restrictions'] or time_info['has_time_restriction'],
                    'summary': self._create_access_summary(category, access_info, time_info)
                })
            
            else:
                # Environmental factors and general notes
                parsed_data.update({
                    'raw_details': details,
                    'parsed_successfully': bool(details.strip()),
                    'summary': f"{category}: {details[:50]}..." if len(details) > 50 else f"{category}: {details}"
                })
        
        except Exception as e:
            print(f"WARNING: Failed to parse {category} constraint details '{details}': {e}")
            parsed_data['error'] = str(e)
        
        return parsed_data

    def _create_availability_summary(self, date_info: Dict, time_info: Dict) -> str:
        """Create human-readable summary of availability constraints"""
        summary_parts = []
        
        if date_info.get('start_date') and date_info.get('end_date'):
            summary_parts.append(f"Available {date_info['start_date']} to {date_info['end_date']}")
        elif date_info.get('available_dates'):
            summary_parts.append(f"Available on specific dates: {', '.join(date_info['available_dates'][:3])}")
        elif date_info.get('restricted_dates'):
            summary_parts.append(f"Restricted dates: {', '.join(date_info['restricted_dates'][:3])}")
        
        if time_info.get('start_time') and time_info.get('end_time'):
            summary_parts.append(f"Hours: {time_info['start_time']}-{time_info['end_time']}")
            if time_info.get('daily_hours'):
                summary_parts.append(f"({time_info['daily_hours']:.1f}h daily)")
        
        return "; ".join(summary_parts) if summary_parts else "Availability constraint"


    def _create_access_summary(self, category: str, access_info: Dict, time_info: Dict) -> str:
        """Create human-readable summary of access restrictions"""
        summary_parts = [category]
        
        if category == 'Sound' and access_info.get('sound_issues'):
            summary_parts.append(f"Issues: {', '.join(access_info['sound_issues'][:2])}")
        elif category == 'Power' and access_info.get('power_limitations'):
            summary_parts.append(f"Limitations: {', '.join(access_info['power_limitations'][:2])}")
        elif category == 'Parking' and access_info.get('parking_restrictions'):
            summary_parts.append(f"Restrictions: {', '.join(access_info['parking_restrictions'][:2])}")
        
        if time_info.get('restricted_times'):
            summary_parts.append(f"Time restrictions: {len(time_info['restricted_times'])} periods")
        
        return "; ".join(summary_parts)


    def _print_location_parsing_stats(self):
        """Print location constraint parsing statistics"""
        stats = self.location_parsing_stats
        
        print(f"\n" + "="*60)
        print(f"STEP 2.5c PHASE A: LOCATION CONSTRAINT PARSING STATS")
        print(f"="*60)
        print(f"📍 Total location constraints processed: {stats['total_location_constraints']}")
        print(f"✅ Availability windows parsed: {stats['availability_windows_parsed']}")
        print(f"🚫 Access restrictions parsed: {stats['access_restrictions_parsed']}")  
        print(f"🌤️ Environmental factors parsed: {stats['environmental_factors_parsed']}")
        print(f"❌ Parsing failures: {stats['parsing_failures']}")
        
        if stats['total_location_constraints'] > 0:
            success_rate = ((stats['total_location_constraints'] - stats['parsing_failures']) / 
                        stats['total_location_constraints']) * 100
            print(f"📊 Success rate: {success_rate:.1f}%")
        
        print(f"\n📋 Constraint categories:")
        for category, count in stats['constraint_categories'].items():
            if count > 0:
                print(f"  • {category}: {count}")
        
        print(f"="*60)

    

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
        """Group scenes by geographic location with REAL time estimation - REDUCED LOGGING"""
        location_groups = defaultdict(list)

        for scene in self.stripboard:
            location = scene.get('Geographic_Location', 'Unknown Location')
            # Skip "Location TBD" or empty locations
            if location and location != 'Location TBD':
                location_groups[location].append(scene)

        clusters = []
        for location, scenes in location_groups.items():
            # Use REAL scene time estimates
            total_hours = 0.0
            for scene in scenes:
                scene_number = scene.get('Scene_Number', '')

                # Get real time estimate or use fallback - try multiple formats
                if str(scene_number) in self.scene_time_estimates:
                    scene_hours = self.scene_time_estimates[str(scene_number)]
                    # REMOVED: Debug print for every scene
                elif scene_number in self.scene_time_estimates:
                    scene_hours = self.scene_time_estimates[scene_number]
                    # REMOVED: Debug print for every scene
                else:
                    # Fallback to page count estimation
                    scene_hours = self._estimate_scene_hours_from_page_count(scene)
                    # REMOVED: Debug print for every scene

                total_hours += scene_hours
        
            # Convert to shooting days using DAILY HOUR LIMITS (10 hours per day)
            MAX_HOURS_PER_DAY = 10.0
            estimated_days = max(1, int((total_hours + MAX_HOURS_PER_DAY - 0.1) / MAX_HOURS_PER_DAY))
            
            # Still limit clusters to reasonable sizes (max 4 days per location)
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
                total_hours=total_hours,
                estimated_days=estimated_days,
                required_actors=list(all_actors)
            ))

        # Sort clusters by total hours (larger first for better scheduling)
        clusters.sort(key=lambda x: x.total_hours, reverse=True)
        
        # SINGLE summary log instead of per-cluster logging
        print(f"DEBUG: Created {len(clusters)} location clusters using REAL time estimates")
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
        """Build efficient lookup structures for constraints - WITH COMPREHENSIVE ERROR HANDLING"""
        # Initialize all data structures with error handling
        try:
            self.actor_unavailable_dates = defaultdict(list)
            self.actor_available_weeks = {}
            self.actor_required_days = {}
            self.actor_constraint_types = {}
            self.actor_constraint_levels = {}
            
            # Director mandate data structures
            self.director_shoot_first = []
            self.director_shoot_last = []
            self.director_sequence_rules = []
            self.director_same_day_groups = []
            self.director_location_groupings = {}
            self.director_mandates_raw = []
            
            # Other constraint maps
            self.location_windows = {}
            
            # Error tracking
            constraint_processing_errors = 0
            constraints_processed = 0
            structured_processed = 0
            fallback_processed = 0
            
            # Process all constraints with individual error handling
            for constraint in self.constraints:
                try:
                    constraints_processed += 1
                    
                    if constraint.actor_restriction:
                        self._map_actor_constraint(constraint)
                    elif constraint.source == ConstraintPriority.DIRECTOR and constraint.affected_scenes:
                        # Track which processing method is used
                        if (constraint.date_restriction and 
                            constraint.date_restriction.get('source_format') == 'structured_v2'):
                            structured_processed += 1
                        else:
                            fallback_processed += 1
                        
                        self._parse_director_constraint_safe(constraint)
                    elif constraint.location_restriction and constraint.location_restriction.get('location'):
                        location = constraint.location_restriction['location']
                        self.location_windows[location] = constraint
                
                except Exception as e:
                    constraint_processing_errors += 1
                    print(f"ERROR: Failed to process constraint: {e}")
                    print(f"       Constraint: {getattr(constraint, 'description', 'Unknown')}")
                    continue
            
            # Build travel times with error handling
            try:
                self._build_travel_times()
            except Exception as e:
                print(f"ERROR: Travel times building failed: {e}")
            
            # Enhanced summary with processing method breakdown and error reporting
            print(f"DEBUG: Processed {constraints_processed} constraints with {constraint_processing_errors} errors")
            print(f"DEBUG: Built constraint maps - {len(self.actor_unavailable_dates)} actors, "
                f"{len(self.director_shoot_first)} 'shoot first', "
                f"{len(self.director_shoot_last)} 'shoot last', "
                f"{len(self.director_sequence_rules)} sequence rules, "
                f"{len(self.director_same_day_groups)} same day groups")
            
            if structured_processed > 0 or fallback_processed > 0:
                print(f"DEBUG: Director constraint processing - {structured_processed} structured (direct mapping), "
                    f"{fallback_processed} fallback (keyword detection)")
            
            # Detailed constraint breakdown for debugging
            if self.director_sequence_rules:
                before_rules = len([r for r in self.director_sequence_rules if r.get('type') == 'before'])
                after_rules = len([r for r in self.director_sequence_rules if r.get('type') == 'after'])
                generic_rules = len([r for r in self.director_sequence_rules if r.get('type') == 'sequence'])
                print(f"DEBUG: Sequence rules breakdown - {before_rules} before, {after_rules} after, {generic_rules} generic")
            
            if self.director_same_day_groups:
                consecutive_groups = len([g for g in self.director_same_day_groups if g.get('type') == 'consecutive'])
                same_day_groups = len(self.director_same_day_groups) - consecutive_groups
                print(f"DEBUG: Day grouping breakdown - {same_day_groups} same day, {consecutive_groups} consecutive days")

        except Exception as e:
            print(f"ERROR: Constraint maps building failed: {e}")
            import traceback
            traceback.print_exc()

    
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
        """Calculate fitness using graduated penalty system - PERFORMANCE OPTIMIZED"""
        score = 0.0
        
        sequence = individual['sequence']
        day_assignments = individual['day_assignments']
        
        # 1. Actor constraint violations (Step 2.5a) - NO LOGGING
        actor_violations = self._check_complete_actor_violations(sequence, day_assignments)
        actor_penalty = PENALTY_HARD_CONSTRAINT * actor_violations
        score += actor_penalty
        
        # 2. Other penalties (unchanged for now)
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
        
        # REMOVED: All debug prints that run thousands of times
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
        
        #if actor_violations > 0:
            #print(f"DEBUG: Found {actor_violations} actor constraint violations")
        
    def _map_actor_constraint(self, constraint):
        """Map complete actor constraint data from n8n - NO DEBUG LOGGING"""
        actor_restriction = constraint.actor_restriction
        actor = actor_restriction.get('actor')
        
        if not actor:
            return
        
        # Store constraint metadata
        self.actor_constraint_levels[actor] = constraint.type.value
        
        # Unavailable dates (NO DEBUG PRINTS)
        unavailable_date = actor_restriction.get('unavailable_date')
        if unavailable_date:
            self.actor_unavailable_dates[actor].append(unavailable_date)
        
        # Available weeks (NO DEBUG PRINTS)
        available_weeks = actor_restriction.get('available_weeks')
        if available_weeks:
            self.actor_available_weeks[actor] = available_weeks
        
        # Required days (NO DEBUG PRINTS)
        required_days = actor_restriction.get('required_days')
        if required_days:
            self.actor_required_days[actor] = required_days 

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

    def _get_shooting_week_from_day(self, day_index: int) -> int:
        """NEW: Convert shooting day index to week number (1-based)"""
        if day_index < 0:
            return 1
        
        # Assuming 6-day weeks (Mon-Sat), with Sundays off
        # Week 1 = days 0-5, Week 2 = days 6-11, etc.
        return (day_index // 6) + 1

    def _check_actor_unavailable_dates(self, sequence: List[int], day_assignments: List[int]) -> int:
        """Check unavailable dates using n8n cast_mapping - MINIMAL LOGGING"""
        violations = 0
        first_violation_logged = False  # NEW: Only log first violation
        
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
                                    # REDUCED: Only log first violation per run to avoid spam
                                    if not first_violation_logged:
                                        #print(f"DEBUG: First actor date violation: {actor} unavailable {shooting_date}")
                                        first_violation_logged = True
                            except:
                                pass
        
        return violations    

    def _check_actor_available_weeks(self, sequence: List[int], day_assignments: List[int]) -> int:
        """Check available weeks using n8n cast_mapping - MINIMAL LOGGING"""
        violations = 0
        first_violation_logged = False  # NEW: Only log first violation
        
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
                                # REDUCED: Only log first violation per run to avoid spam
                                if not first_violation_logged:
                                    #print(f"DEBUG: First actor week violation: {actor} in week {day_week}, needs {available_weeks}")
                                    first_violation_logged = True
                                break
        
        return violations

    def _check_actor_required_days(self, sequence: List[int], day_assignments: List[int]) -> int:
        """Check actor days using n8n cast_mapping - REDUCED LOGGING"""
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
        
        # REMOVED: Excessive debug logging of all character scheduled days
        
        # Check against required days (actor constraints) - ONLY LOG VIOLATIONS
        for actor, required_days in self.actor_required_days.items():
            # Find character(s) this actor plays
            characters_for_actor = [char for char, mapped_actor in self.cast_mapping.items() 
                                if mapped_actor == actor]
            
            total_scheduled_days = 0
            for character in characters_for_actor:
                total_scheduled_days += character_scheduled_days.get(character, 0)
            
            if total_scheduled_days != required_days:
                violations += 1
                # ONLY LOG ACTUAL VIOLATIONS (not success cases)
                #print(f"DEBUG: VIOLATION - Actor '{actor}' scheduled {total_scheduled_days} days, needs {required_days}")
        
        return violations

    def _get_actor_for_character(self, character_name: str) -> Optional[str]:
        """Get actor name from character name using n8n cast_mapping - NO LOGGING"""
        # REMOVED: Debug logging that prints thousands of times during GA
        return self.cast_mapping.get(character_name)


    def _count_location_splits(self, sequence: List[int], day_assignments: List[int]) -> int:
        """Count how many locations are split across multiple non-consecutive days"""
        # In this implementation, each cluster is assigned to consecutive days
        # so we don't have location splits. This is more of a penalty for
        # future algorithms that might split clusters.
        return 0
    
    def _count_director_violations(self, sequence: List[int], day_assignments: List[int]) -> int:
        """Count violations of director mandates - PHASE B IMPLEMENTATION"""
        return self._check_complete_director_violations(sequence, day_assignments)

    def _parse_director_constraint(self, constraint):
        """Parse director constraint - PHASE B: Direct mapping for structured data"""
        
        affected_scenes = constraint.affected_scenes or []
        constraint_level = constraint.type.value
        
        # PHASE B: Check if we have structured data from new two-agent system
        if (constraint.date_restriction and 
            constraint.date_restriction.get('source_format') == 'structured_v2'):
            
            # NEW: Direct mapping - no keyword detection needed!
            structured_type = constraint.date_restriction.get('constraint_type', '')
            reasoning = constraint.date_restriction.get('reasoning', '')
            locations = constraint.date_restriction.get('locations', [])
            
            print(f"DEBUG: Processing structured director constraint: '{structured_type}' for scenes {affected_scenes}")
            
            # Direct constraint type mapping (fast and reliable)
            if structured_type == 'shoot_first':
                self.director_shoot_first.extend(affected_scenes)
                print(f"DEBUG: Added 'shoot first' mandate for scenes: {affected_scenes}")
            
            elif structured_type == 'shoot_last':
                self.director_shoot_last.extend(affected_scenes)
                print(f"DEBUG: Added 'shoot last' mandate for scenes: {affected_scenes}")
            
            elif structured_type == 'sequence_before_after':
                sequence_rule = self._create_sequence_rule('before', affected_scenes, constraint_level, reasoning)
                if sequence_rule:
                    self.director_sequence_rules.append(sequence_rule)
                    print(f"DEBUG: Added 'before' sequence rule: scenes {affected_scenes}")
            
            elif structured_type == 'sequence_after_before':
                sequence_rule = self._create_sequence_rule('after', affected_scenes, constraint_level, reasoning)
                if sequence_rule:
                    self.director_sequence_rules.append(sequence_rule)
                    print(f"DEBUG: Added 'after' sequence rule: scenes {affected_scenes}")
            
            elif structured_type == 'same_day_grouping':
                if len(affected_scenes) > 1:
                    self.director_same_day_groups.append({
                        'scenes': affected_scenes,
                        'level': constraint_level,
                        'reasoning': reasoning,
                        'constraint_text': constraint.description
                    })
                    print(f"DEBUG: Added same day grouping for scenes: {affected_scenes}")
            
            elif structured_type == 'consecutive_days':
                if len(affected_scenes) > 1:
                    self.director_same_day_groups.append({
                        'scenes': affected_scenes,
                        'level': constraint_level,
                        'reasoning': reasoning,
                        'constraint_text': constraint.description,
                        'type': 'consecutive'  # Special marker for consecutive vs same day
                    })
                    print(f"DEBUG: Added consecutive days grouping for scenes: {affected_scenes}")
            
            elif structured_type == 'location_grouping':
                self._handle_location_grouping_structured(constraint, locations, constraint_level, reasoning)
                print(f"DEBUG: Added location grouping for locations: {locations}")
            
            elif structured_type in ['actor_rest_day', 'prep_time_required', 'wrap_time_required']:
                # Future constraint types - placeholder for Phase 3
                print(f"DEBUG: Structured constraint type '{structured_type}' recognized but not yet implemented")
            
            else:
                print(f"DEBUG: Unknown structured constraint type: '{structured_type}' - will use fallback parsing")
                self._parse_director_constraint_fallback(constraint)
        
        else:
            # FALLBACK: Use legacy keyword detection for old formats
            print(f"DEBUG: Using fallback parsing for legacy constraint: '{constraint.description}'")
            self._parse_director_constraint_fallback(constraint)

    def _create_sequence_rule(self, rule_type: str, affected_scenes: List[str], 
                         constraint_level: str, reasoning: str) -> dict:
        """Create sequence rule from structured data"""
        
        if len(affected_scenes) < 2:
            print(f"DEBUG: Sequence rule needs at least 2 scenes, got: {affected_scenes}")
            return None
        
        sequence_rule = {
            'type': rule_type,  # 'before' or 'after'
            'first_scene': str(affected_scenes[0]),
            'second_scene': str(affected_scenes[1]),
            'level': constraint_level,
            'reasoning': reasoning,
            'all_scenes': affected_scenes  # Store all scenes if more than 2
        }

        # Handle multiple scenes in sequence
        if len(affected_scenes) > 2:
            sequence_rule['additional_scenes'] = [str(s) for s in affected_scenes[2:]]
        
        return sequence_rule


    def _handle_location_grouping_structured(self, constraint, locations: List[str], 
                                       constraint_level: str, reasoning: str):
        """Handle location grouping from structured data"""
        
        for location in locations:
            if location not in self.director_location_groupings:
                self.director_location_groupings[location] = []
            
            self.director_location_groupings[location].append({
                'constraint': constraint,
                'requirement_type': 'grouping',
                'level': constraint_level,
                'reasoning': reasoning,
                'source': 'structured_v2'
            })
    

    def _parse_director_constraint_fallback(self, constraint):
        """Fallback parsing for legacy constraint formats"""
        
        constraint_text = constraint.description.lower() if constraint.description else ""
        affected_scenes = constraint.affected_scenes or []
        constraint_level = constraint.type.value
        
        # Simplified keyword detection (keep essential logic only)
        if self._is_shoot_first_constraint(constraint_text):
            self.director_shoot_first.extend(affected_scenes)
            print(f"DEBUG: Fallback detected 'shoot first' for scenes: {affected_scenes}")
        
        elif self._is_shoot_last_constraint(constraint_text):
            self.director_shoot_last.extend(affected_scenes)
            print(f"DEBUG: Fallback detected 'shoot last' for scenes: {affected_scenes}")
        
        elif self._is_sequence_constraint(constraint_text):
            sequence_rule = self._parse_sequence_rule_fallback(constraint_text, affected_scenes, constraint_level)
            if sequence_rule:
                self.director_sequence_rules.append(sequence_rule)
                print(f"DEBUG: Fallback detected sequence rule: {sequence_rule}")
        
        elif self._is_same_day_constraint(constraint_text):
            if len(affected_scenes) > 1:
                self.director_same_day_groups.append({
                    'scenes': affected_scenes,
                    'level': constraint_level,
                    'constraint_text': constraint.description,
                    'source': 'fallback'
                })
                print(f"DEBUG: Fallback detected same day grouping for scenes: {affected_scenes}")
        
        elif self._is_location_grouping_constraint(constraint_text, constraint):
            self._parse_location_grouping_fallback(constraint)
            print(f"DEBUG: Fallback detected location grouping constraint")
        
        else:
            print(f"DEBUG: Fallback could not categorize constraint: '{constraint.description}'")
    
    def _is_shoot_first_constraint(self, text: str) -> bool:
        """Detect 'shoot first' type constraints"""
        first_keywords = ["shoot first", "start with", "begin with", "opening", "first day", "early"]
        return any(keyword in text for keyword in first_keywords)

    def _is_shoot_last_constraint(self, text: str) -> bool:
        """Detect 'shoot last' type constraints"""
        last_keywords = ["shoot last", "end with", "finish with", "closing", "final", "last day", "wrap"]
        return any(keyword in text for keyword in last_keywords)

    def _is_sequence_constraint(self, text: str) -> bool:
        """Detect sequence order constraints"""
        sequence_keywords = ["before", "after", "must follow", "sequence", "order", "then"]
        return any(keyword in text for keyword in sequence_keywords)

    def _is_same_day_constraint(self, text: str) -> bool:
        """Detect same day grouping constraints"""
        same_day_keywords = ["same day", "together", "consecutive", "back to back", "one day"]
        return any(keyword in text for keyword in same_day_keywords)

    def _is_location_grouping_constraint(self, text: str, constraint) -> bool:
        """Detect location-specific grouping constraints"""
        location_keywords = ["location", "all scenes at", "group by location"]
        has_location_data = (constraint.date_restriction and 
                            'locations' in constraint.date_restriction)
        return any(keyword in text for keyword in location_keywords) or has_location_data

    def _parse_sequence_rule_fallback(self, text: str, scenes: List[str], level: str) -> dict:
        """Simplified fallback sequence rule parsing"""
        
        if len(scenes) < 2:
            return None
        
        # Simple before/after detection
        if "before" in text:
            return {
                'type': 'before',
                'first_scene': str(scenes[0]),
                'second_scene': str(scenes[1]),
                'level': level,
                'source': 'fallback_keyword'
            }
        elif "after" in text:
            return {
                'type': 'after',
                'first_scene': str(scenes[0]),
                'second_scene': str(scenes[1]),
                'level': level,
                'source': 'fallback_keyword'
            }
        else:
            # Generic sequence requirement
            return {
                'type': 'sequence',
                'scenes': [str(s) for s in scenes],
                'level': level,
                'source': 'fallback_generic'
            }

    def _parse_location_grouping_fallback(self, constraint):
        """Simplified fallback location grouping parsing"""
        
        # Extract locations from constraint
        locations = []
        if constraint.date_restriction and 'locations' in constraint.date_restriction:
            locations = constraint.date_restriction['locations']
        
        # Store location grouping requirements
        for location in locations:
            if location not in self.director_location_groupings:
                self.director_location_groupings[location] = []
            
            self.director_location_groupings[location].append({
                'constraint': constraint,
                'requirement_type': 'grouping',
                'level': constraint.type.value,
                'source': 'fallback'
            })

    def _check_complete_director_violations(self, sequence: List[int], day_assignments: List[int]) -> int:
        """Check ALL director mandate violations - Phase B Implementation"""
        violations = 0
        
        # Build scene-to-day mapping for efficient lookups
        scene_to_day = self._build_scene_to_day_mapping(sequence, day_assignments)
        
        # Check all director constraint types
        violations += self._check_shoot_first_violations(scene_to_day)
        violations += self._check_shoot_last_violations(scene_to_day)
        violations += self._check_sequence_rule_violations(scene_to_day)
        violations += self._check_same_day_violations(scene_to_day)
        violations += self._check_location_grouping_violations(sequence, day_assignments)
        
        return violations

    def _check_shoot_first_violations(self, scene_to_day: Dict[str, int]) -> int:
        """Check if 'shoot first' scenes are scheduled early - WITH COMPREHENSIVE ERROR HANDLING"""
        violations = 0
        
        try:
            if not self.director_shoot_first:
                return 0
            
            # Validate scene_to_day mapping
            if not isinstance(scene_to_day, dict):
                print(f"ERROR: Invalid scene_to_day type: {type(scene_to_day)}")
                return 0
            
            # Find earliest day among mandated scenes with validation
            earliest_mandated_day = float('inf')
            valid_mandated_scenes = 0
            
            for scene_num in self.director_shoot_first:
                try:
                    if scene_num in scene_to_day:
                        day_value = scene_to_day[scene_num]
                        if isinstance(day_value, (int, float)) and day_value >= 0:
                            earliest_mandated_day = min(earliest_mandated_day, day_value)
                            valid_mandated_scenes += 1
                        else:
                            print(f"WARNING: Invalid day value for scene {scene_num}: {day_value}")
                    else:
                        print(f"WARNING: Shoot first scene {scene_num} not found in schedule")
                except Exception as e:
                    print(f"ERROR: Processing shoot first scene {scene_num}: {e}")
            
            if valid_mandated_scenes == 0:
                print(f"WARNING: No valid shoot first scenes found in schedule")
                return 0
            
            # Count non-mandated scenes scheduled before earliest mandated scene
            if earliest_mandated_day != float('inf'):
                for scene_num, day in scene_to_day.items():
                    try:
                        if (scene_num not in self.director_shoot_first and 
                            isinstance(day, (int, float)) and 
                            day < earliest_mandated_day):
                            violations += 1
                    except Exception as e:
                        print(f"ERROR: Comparing scene {scene_num} day {day}: {e}")
        
        except Exception as e:
            print(f"ERROR: Shoot first violations check failed: {e}")
            return 0
        
        return violations

    def _check_shoot_last_violations(self, scene_to_day: Dict[str, int]) -> int:
        """Check if 'shoot last' scenes are scheduled late - WITH COMPREHENSIVE ERROR HANDLING"""
        violations = 0
        
        try:
            if not self.director_shoot_last:
                return 0
            
            # Validate scene_to_day mapping
            if not isinstance(scene_to_day, dict):
                print(f"ERROR: Invalid scene_to_day type: {type(scene_to_day)}")
                return 0
            
            # Find latest day among mandated scenes with validation
            latest_mandated_day = -1
            valid_mandated_scenes = 0
            
            for scene_num in self.director_shoot_last:
                try:
                    if scene_num in scene_to_day:
                        day_value = scene_to_day[scene_num]
                        if isinstance(day_value, (int, float)) and day_value >= 0:
                            latest_mandated_day = max(latest_mandated_day, day_value)
                            valid_mandated_scenes += 1
                        else:
                            print(f"WARNING: Invalid day value for scene {scene_num}: {day_value}")
                    else:
                        print(f"WARNING: Shoot last scene {scene_num} not found in schedule")
                except Exception as e:
                    print(f"ERROR: Processing shoot last scene {scene_num}: {e}")
            
            if valid_mandated_scenes == 0:
                print(f"WARNING: No valid shoot last scenes found in schedule")
                return 0
            
            # Count non-mandated scenes scheduled after latest mandated scene
            if latest_mandated_day >= 0:
                for scene_num, day in scene_to_day.items():
                    try:
                        if (scene_num not in self.director_shoot_last and 
                            isinstance(day, (int, float)) and 
                            day > latest_mandated_day):
                            violations += 1
                    except Exception as e:
                        print(f"ERROR: Comparing scene {scene_num} day {day}: {e}")
        
        except Exception as e:
            print(f"ERROR: Shoot last violations check failed: {e}")
            return 0
        
        return violations

    def _check_sequence_rule_violations(self, scene_to_day: Dict[str, int]) -> int:
        """Check sequence rule violations - WITH COMPREHENSIVE ERROR HANDLING"""
        violations = 0
        
        try:
            if not isinstance(scene_to_day, dict):
                print(f"ERROR: Invalid scene_to_day type: {type(scene_to_day)}")
                return 0
            
            for rule in self.director_sequence_rules:
                try:
                    if not isinstance(rule, dict):
                        print(f"WARNING: Invalid sequence rule type: {type(rule)}")
                        continue
                    
                    rule_type = rule.get('type')
                    first_scene = rule.get('first_scene')
                    second_scene = rule.get('second_scene')
                    
                    # Validate rule data
                    if not all([rule_type, first_scene, second_scene]):
                        print(f"WARNING: Incomplete sequence rule: {rule}")
                        continue
                    
                    # Convert to strings for consistency
                    first_scene = str(first_scene).strip()
                    second_scene = str(second_scene).strip()
                    
                    if rule_type == 'before':
                        if (first_scene in scene_to_day and second_scene in scene_to_day):
                            try:
                                first_day = scene_to_day[first_scene]
                                second_day = scene_to_day[second_scene]
                                
                                if (isinstance(first_day, (int, float)) and 
                                    isinstance(second_day, (int, float)) and
                                    first_day >= second_day):
                                    violations += 1
                            except Exception as e:
                                print(f"ERROR: Comparing days for scenes {first_scene}, {second_scene}: {e}")
                    
                    elif rule_type == 'after':
                        if (first_scene in scene_to_day and second_scene in scene_to_day):
                            try:
                                first_day = scene_to_day[first_scene]
                                second_day = scene_to_day[second_scene]
                                
                                if (isinstance(first_day, (int, float)) and 
                                    isinstance(second_day, (int, float)) and
                                    first_day <= second_day):
                                    violations += 1
                            except Exception as e:
                                print(f"ERROR: Comparing days for scenes {first_scene}, {second_scene}: {e}")
                    
                    else:
                        print(f"WARNING: Unknown sequence rule type: {rule_type}")
                
                except Exception as e:
                    print(f"ERROR: Processing sequence rule {rule}: {e}")
                    continue
        
        except Exception as e:
            print(f"ERROR: Sequence rule violations check failed: {e}")
            return 0
        
        return violations

    def _check_same_day_violations(self, scene_to_day: Dict[str, int]) -> int:
        """Check same day grouping violations - WITH COMPREHENSIVE ERROR HANDLING"""
        violations = 0
        
        try:
            if not isinstance(scene_to_day, dict):
                print(f"ERROR: Invalid scene_to_day type: {type(scene_to_day)}")
                return 0
            
            for group in self.director_same_day_groups:
                try:
                    if not isinstance(group, dict):
                        print(f"WARNING: Invalid same day group type: {type(group)}")
                        continue
                    
                    scenes = group.get('scenes', [])
                    if not isinstance(scenes, list):
                        print(f"WARNING: Invalid scenes list in same day group: {type(scenes)}")
                        continue
                    
                    # Get days for all scenes in group with validation
                    group_days = []
                    valid_scenes = []
                    
                    for scene_num in scenes:
                        try:
                            scene_str = str(scene_num).strip()
                            if scene_str in scene_to_day:
                                day_value = scene_to_day[scene_str]
                                if isinstance(day_value, (int, float)) and day_value >= 0:
                                    group_days.append(day_value)
                                    valid_scenes.append(scene_str)
                                else:
                                    print(f"WARNING: Invalid day value for scene {scene_str}: {day_value}")
                            else:
                                print(f"WARNING: Same day group scene {scene_str} not found in schedule")
                        except Exception as e:
                            print(f"ERROR: Processing same day group scene {scene_num}: {e}")
                    
                    # Check if scenes are spread across multiple days
                    if len(valid_scenes) > 1 and len(set(group_days)) > 1:
                        violations += len(group_days) - 1  # Penalty proportional to spread
                
                except Exception as e:
                    print(f"ERROR: Processing same day group {group}: {e}")
                    continue
        
        except Exception as e:
            print(f"ERROR: Same day violations check failed: {e}")
            return 0
    
        return violations

    def _check_location_grouping_violations(self, sequence: List[int], day_assignments: List[int]) -> int:
        """Check location grouping violations"""
        violations = 0
        
        # Check if location shoots are consecutive
        for location, requirements in self.director_location_groupings.items():
            # Find all clusters for this location
            location_days = []
            for i, cluster_idx in enumerate(sequence):
                cluster = self.cluster_manager.clusters[cluster_idx]
                if cluster.location == location:
                    location_days.append(day_assignments[i])
            
            # Check if location shoots are consecutive
            if len(location_days) > 1:
                location_days.sort()
                for j in range(len(location_days) - 1):
                    if location_days[j + 1] - location_days[j] > 1:
                        violations += 1  # Gap between location shoots
        
        return violations




    def _build_scene_to_day_mapping(self, sequence: List[int], day_assignments: List[int]) -> Dict[str, int]:
        """Build efficient scene number to shooting day mapping - WITH COMPREHENSIVE VALIDATION"""
        scene_to_day = {}
        
        try:
            # Validate inputs
            if not isinstance(sequence, list) or not isinstance(day_assignments, list):
                print(f"ERROR: Invalid mapping inputs - sequence: {type(sequence)}, assignments: {type(day_assignments)}")
                return {}
            
            if len(sequence) != len(day_assignments):
                print(f"ERROR: Sequence length {len(sequence)} != assignments length {len(day_assignments)}")
                return {}
            
            for i, cluster_idx in enumerate(sequence):
                try:
                    # Validate cluster index
                    if not isinstance(cluster_idx, int) or cluster_idx < 0:
                        print(f"WARNING: Invalid cluster index {cluster_idx} at position {i}")
                        continue
                    
                    if cluster_idx >= len(self.cluster_manager.clusters):
                        print(f"WARNING: Cluster index {cluster_idx} out of range (max: {len(self.cluster_manager.clusters)-1})")
                        continue
                    
                    cluster = self.cluster_manager.clusters[cluster_idx]
                    start_day = day_assignments[i]
                    
                    # Validate start day
                    if not isinstance(start_day, (int, float)) or start_day < 0:
                        print(f"WARNING: Invalid start day {start_day} for cluster {cluster_idx}")
                        continue
                    
                    # Map each scene in this cluster to its shooting day
                    for scene in cluster.scenes:
                        try:
                            scene_number = scene.get('Scene_Number')
                            if scene_number is not None:
                                scene_number_str = str(scene_number).strip()
                                if scene_number_str:
                                    scene_to_day[scene_number_str] = int(start_day)
                        except Exception as e:
                            print(f"WARNING: Failed to map scene {scene.get('Scene_Number', 'Unknown')}: {e}")
                
                except Exception as e:
                    print(f"ERROR: Processing cluster {cluster_idx}: {e}")
                    continue
        
        except Exception as e:
            print(f"ERROR: Scene-to-day mapping failed: {e}")
        
        return scene_to_day

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
        """Run genetic algorithm with proper fitness tracking - FIXED FORMAT"""
        pop_size = self.params.get('phase1_population', 50)
        generations = self.params.get('phase1_generations', 200)
        
        # Initialize population
        population = [self.create_individual() for _ in range(pop_size)]
        best_individual = None
        best_fitness = -float('inf')
        
        # TRACK FITNESS IMPROVEMENT
        fitness_history = []
        
        print(f"DEBUG: Starting evolution with {pop_size} individuals for {generations} generations")
        
        for gen in range(generations):
            # Evaluate fitness
            fitnesses = []
            for ind in population:
                fitness = self.fitness(ind)
                fitnesses.append(fitness)
            
            # Track best
            gen_best_idx = np.argmax(fitnesses)
            gen_best_fitness = fitnesses[gen_best_idx]
            
            if gen_best_fitness > best_fitness:
                best_fitness = gen_best_fitness
                best_individual = population[gen_best_idx].copy()
            
            # RECORD FITNESS PROGRESS
            avg_fitness = sum(fitnesses) / len(fitnesses)
            fitness_history.append({
                'generation': gen,
                'best_fitness': gen_best_fitness,
                'avg_fitness': avg_fitness,
                'worst_fitness': min(fitnesses)
            })
            
            # CORRECTED: Progress logging every 50 generations with IMPROVEMENT tracking
            if gen % 50 == 0 or gen == generations - 1:
                improvement = gen_best_fitness - fitness_history[0]['best_fitness'] if fitness_history else 0
                print(f"DEBUG: Gen {gen}, Best: {gen_best_fitness:.0f}, Avg: {avg_fitness:.0f}, Improvement: +{improvement:.0f}")
            
            # Create new population
            new_population = []
            
            # Elitism - keep best individuals
            elite_count = max(2, pop_size // 10)
            elite_indices = np.argsort(fitnesses)[-elite_count:]
            for idx in elite_indices:
                new_population.append(self._copy_individual(population[idx]))
            
            # Generate rest through selection and reproduction
            while len(new_population) < pop_size:
                parent1 = self._tournament_selection(population, fitnesses)
                parent2 = self._tournament_selection(population, fitnesses)
                child = self._crossover(parent1, parent2)
                child = self._mutate(child)
                new_population.append(child)
            
            population = new_population[:pop_size]
        
        # FINAL SUMMARY
        total_improvement = best_fitness - fitness_history[0]['best_fitness'] if fitness_history else 0
        print(f"DEBUG: Evolution completed. Final fitness: {best_fitness:.0f}, Total improvement: +{total_improvement:.0f}")
        
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

    def _parse_director_constraint_safe(self, constraint):
        """Parse director constraint with comprehensive error handling - NEW SAFE VERSION"""
        
        try:
            affected_scenes = constraint.affected_scenes or []
            constraint_level = constraint.type.value
            
            # Check if we have structured data from new two-agent system
            if (constraint.date_restriction and 
                constraint.date_restriction.get('source_format') == 'structured_v2'):
                
                # Direct mapping with error handling - no keyword detection needed!
                structured_type = constraint.date_restriction.get('constraint_type', '')
                reasoning = constraint.date_restriction.get('reasoning', '')
                locations = constraint.date_restriction.get('locations', [])
                
                print(f"DEBUG: Processing structured director constraint: '{structured_type}' for scenes {affected_scenes}")
                
                try:
                    # Direct constraint type mapping with validation
                    if structured_type == 'shoot_first':
                        valid_scenes = [str(s).strip() for s in affected_scenes if s is not None and str(s).strip()]
                        self.director_shoot_first.extend(valid_scenes)
                        print(f"DEBUG: Added 'shoot first' mandate for scenes: {valid_scenes}")
                    
                    elif structured_type == 'shoot_last':
                        valid_scenes = [str(s).strip() for s in affected_scenes if s is not None and str(s).strip()]
                        self.director_shoot_last.extend(valid_scenes)
                        print(f"DEBUG: Added 'shoot last' mandate for scenes: {valid_scenes}")
                    
                    elif structured_type == 'sequence_before_after':
                        sequence_rule = self._create_sequence_rule_safe('before', affected_scenes, constraint_level, reasoning)
                        if sequence_rule:
                            self.director_sequence_rules.append(sequence_rule)
                            print(f"DEBUG: Added 'before' sequence rule: scenes {affected_scenes}")
                    
                    elif structured_type == 'sequence_after_before':
                        sequence_rule = self._create_sequence_rule_safe('after', affected_scenes, constraint_level, reasoning)
                        if sequence_rule:
                            self.director_sequence_rules.append(sequence_rule)
                            print(f"DEBUG: Added 'after' sequence rule: scenes {affected_scenes}")
                    
                    elif structured_type == 'same_day_grouping':
                        if len(affected_scenes) > 1:
                            valid_scenes = [str(s).strip() for s in affected_scenes if s is not None and str(s).strip()]
                            if len(valid_scenes) > 1:
                                self.director_same_day_groups.append({
                                    'scenes': valid_scenes,
                                    'level': constraint_level,
                                    'reasoning': reasoning,
                                    'constraint_text': constraint.description
                                })
                                print(f"DEBUG: Added same day grouping for scenes: {valid_scenes}")
                    
                    elif structured_type == 'consecutive_days':
                        if len(affected_scenes) > 1:
                            valid_scenes = [str(s).strip() for s in affected_scenes if s is not None and str(s).strip()]
                            if len(valid_scenes) > 1:
                                self.director_same_day_groups.append({
                                    'scenes': valid_scenes,
                                    'level': constraint_level,
                                    'reasoning': reasoning,
                                    'constraint_text': constraint.description,
                                    'type': 'consecutive'
                                })
                                print(f"DEBUG: Added consecutive days grouping for scenes: {valid_scenes}")
                    
                    elif structured_type == 'location_grouping':
                        self._handle_location_grouping_safe(constraint, locations, constraint_level, reasoning)
                        print(f"DEBUG: Added location grouping for locations: {locations}")
                    
                    elif structured_type in ['actor_rest_day', 'prep_time_required', 'wrap_time_required']:
                        # Future constraint types - placeholder for Phase 3
                        print(f"DEBUG: Structured constraint type '{structured_type}' recognized but not yet implemented")
                    
                    else:
                        print(f"DEBUG: Unknown structured constraint type: '{structured_type}' - will use fallback parsing")
                        self._parse_director_constraint_fallback(constraint)
                
                except Exception as e:
                    print(f"ERROR: Structured director constraint processing failed: {e}")
                    # Try fallback as recovery
                    self._parse_director_constraint_fallback(constraint)
            
            else:
                # FALLBACK: Use legacy keyword detection for old formats
                print(f"DEBUG: Using fallback parsing for legacy constraint: '{constraint.description}'")
                self._parse_director_constraint_fallback(constraint)
        
        except Exception as e:
            print(f"ERROR: Director constraint parsing failed: {e}")
            print(f"       Constraint: {getattr(constraint, 'description', 'Unknown')}")

    def _create_sequence_rule_safe(self, rule_type: str, affected_scenes: List[str], 
                             constraint_level: str, reasoning: str) -> Optional[dict]:
        """Create sequence rule from structured data with error handling"""
        
        try:
            # Validate inputs
            if not isinstance(affected_scenes, list) or len(affected_scenes) < 2:
                print(f"DEBUG: Sequence rule needs at least 2 scenes, got: {affected_scenes}")
                return None
            
            # Clean and validate scene numbers
            clean_scenes = []
            for scene in affected_scenes:
                try:
                    if scene is not None:
                        scene_str = str(scene).strip()
                        if scene_str:
                            clean_scenes.append(scene_str)
                except Exception as e:
                    print(f"WARNING: Invalid scene in sequence rule '{scene}': {e}")
            
            if len(clean_scenes) < 2:
                print(f"DEBUG: Not enough valid scenes for sequence rule: {clean_scenes}")
                return None
            
            sequence_rule = {
                'type': rule_type,  # 'before' or 'after'
                'first_scene': clean_scenes[0],
                'second_scene': clean_scenes[1],
                'level': constraint_level,
                'reasoning': reasoning,
                'all_scenes': clean_scenes  # Store all scenes if more than 2
            }

            # Handle multiple scenes in sequence
            if len(clean_scenes) > 2:
                sequence_rule['additional_scenes'] = clean_scenes[2:]
            
            return sequence_rule
        
        except Exception as e:
            print(f"ERROR: Failed to create sequence rule: {e}")
            return None
        
        
    def _handle_location_grouping_safe(self, constraint, locations: List[str], 
                                 constraint_level: str, reasoning: str):
        """Handle location grouping from structured data with error handling"""
        
        try:
            # Validate and clean locations
            clean_locations = []
            for location in locations:
                try:
                    if location is not None:
                        location_str = str(location).strip()
                        if location_str:
                            clean_locations.append(location_str)
                except Exception as e:
                    print(f"WARNING: Invalid location in grouping '{location}': {e}")
            
            for location in clean_locations:
                try:
                    if location not in self.director_location_groupings:
                        self.director_location_groupings[location] = []
                    
                    self.director_location_groupings[location].append({
                        'constraint': constraint,
                        'requirement_type': 'grouping',
                        'level': constraint_level,
                        'reasoning': reasoning,
                        'source': 'structured_v2'
                    })
                except Exception as e:
                    print(f"ERROR: Failed to add location grouping for '{location}': {e}")
        
        except Exception as e:
            print(f"ERROR: Location grouping handling failed: {e}")

    
    def validate_constraint_system(self) -> Dict[str, Any]:
        """Validate entire constraint system and return health report"""
        report = {
            'total_constraints': len(self.constraints) if hasattr(self, 'constraints') else 0,
            'valid_constraints': 0,
            'invalid_constraints': 0,
            'errors': [],
            'warnings': [],
            'recommendations': []
        }
        
        try:
            if not hasattr(self, 'constraints'):
                report['errors'].append("No constraints attribute found")
                return report
            
            for i, constraint in enumerate(self.constraints):
                try:
                    # Validate constraint structure
                    if not hasattr(constraint, 'source') or not hasattr(constraint, 'type'):
                        report['errors'].append(f"Constraint {i}: Missing required attributes")
                        report['invalid_constraints'] += 1
                        continue
                    
                    if not hasattr(constraint, 'description') or not constraint.description:
                        report['warnings'].append(f"Constraint {i}: Empty description")
                    
                    if hasattr(constraint, 'affected_scenes'):
                        if constraint.affected_scenes and not isinstance(constraint.affected_scenes, list):
                            report['errors'].append(f"Constraint {i}: affected_scenes not a list")
                            report['invalid_constraints'] += 1
                            continue
                    
                    report['valid_constraints'] += 1
                
                except Exception as e:
                    report['invalid_constraints'] += 1
                    report['errors'].append(f"Constraint {i}: Validation error - {e}")
            
            # Add recommendations
            if report['invalid_constraints'] > 0:
                report['recommendations'].append("Review and fix invalid constraints")
            if len(report['warnings']) > 5:
                report['recommendations'].append("Consider improving constraint data quality")
        
        except Exception as e:
            report['errors'].append(f"Validation system error: {e}")
        
        return report


class ScheduleOptimizer:
    """Main orchestrator for location-first optimization"""
    
    def __init__(self, request: ScheduleRequest):
        """Initialize optimizer - ENHANCED for Phase A director structure handling"""
        self.stripboard = request.stripboard
        self.constraints_raw = request.constraints
        self.params = request.ga_params
        
        # Initialize components
        self.parser = StructuredConstraintParser()
        self.constraints = self.parser.parse_all_constraints(self.constraints_raw)
        
        # Extract cast_mapping from constraints
        self.cast_mapping = self._extract_cast_mapping()
        
        # Create location cluster manager
        scene_time_estimates = self._get_scene_time_estimates()
        self.cluster_manager = LocationClusterManager(self.stripboard, scene_time_estimates)
        
        # Determine shooting calendar
        self.calendar = ShootingCalendar("2025-09-01", "2025-10-31")
        
        # NEW: Analyze director constraint format distribution
        director_constraints = [c for c in self.constraints if c.source == ConstraintPriority.DIRECTOR]
        structured_v2 = len([c for c in director_constraints 
                            if c.date_restriction and c.date_restriction.get('source_format') == 'structured_v2'])
        legacy_v1 = len([c for c in director_constraints 
                        if c.date_restriction and c.date_restriction.get('source_format') == 'legacy_v1'])
        legacy_list = len([c for c in director_constraints 
                        if c.date_restriction and c.date_restriction.get('source_format') == 'legacy_list'])
        
        # Enhanced summary log with format breakdown
        print(f"DEBUG: Initialized optimizer - {len(self.constraints)} total constraints, "
            f"{len(self.cast_mapping)} cast mappings, "
            f"{len(self.cluster_manager.clusters)} clusters, "
            f"{len(self.calendar.shooting_days)} shooting days")
        
        if director_constraints:
            print(f"DEBUG: Director constraints format - {structured_v2} structured_v2, "
                f"{legacy_v1} legacy_v1, {legacy_list} legacy_list")


    def _extract_cast_mapping(self) -> Dict[str, str]:
        """Extract cast_mapping from parsed constraints - NO VERBOSE LOGGING"""
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
                        # REMOVED: Debug print for every character mapping
                
                # SINGLE summary log only
                print(f"DEBUG: Extracted {len(cast_mapping)} character-actor mappings")
                return cast_mapping
        
        print(f"DEBUG: No cast_mapping found in constraints")
        return {}
    
    def optimize(self) -> Dict[str, Any]:
        """Run location-first optimization - WITH CONFLICTS SUMMARY"""
        import time
        start_time = time.time()
        
        # Pass cast_mapping to GA
        ga = LocationFirstGA(self.cluster_manager, self.constraints, self.calendar, 
                            self.params, self.cast_mapping)
        best_individual, best_fitness = ga.evolve()
        
        # Build final schedule
        final_schedule = self._build_final_schedule(best_individual)
        schedule_summary = self._build_schedule_summary(final_schedule)
        metrics = self._calculate_metrics(final_schedule)
        missing_scenes_summary = {
            'total_missing_scenes': len(getattr(self, 'missing_scenes_summary', [])),
            'missing_scenes_details': getattr(self, 'missing_scenes_summary', [])
        }
        
        # Generate detailed conflicts report with summary
        conflicts_data = self._generate_conflicts_report(final_schedule, best_individual)
        
        processing_time = time.time() - start_time
        
        return {
            'schedule': final_schedule,
            'summary': schedule_summary,
            'missing_scenes': missing_scenes_summary,
            'conflicts': conflicts_data['detailed_conflicts'],      # Granular conflicts for UI drill-down
            'conflicts_summary': conflicts_data['conflicts_summary'], # NEW: Constraint-level summary
            'metrics': metrics,
            'fitness_score': best_fitness,
            'processing_time_seconds': processing_time
        }
    
    def _generate_conflicts_report(self, schedule: List[Dict], best_individual: Dict) -> Dict:
        """Generate detailed conflicts AND constraint-level summary"""
        detailed_conflicts = []
        
        try:
            print(f"DEBUG: Generating conflicts report for {len(schedule)} days")
            
            # Convert schedule back to GA format for violation detection
            sequence, day_assignments = self._schedule_to_ga_format(schedule)
            
            if not sequence or not day_assignments:
                print(f"DEBUG: Could not convert schedule to GA format for conflict analysis")
                return {'detailed_conflicts': [], 'conflicts_summary': self._empty_conflicts_summary()}
            
            # Create temporary GA instance for violation detection
            temp_ga = LocationFirstGA(self.cluster_manager, self.constraints, self.calendar, 
                                    self.params, self.cast_mapping)
            
            # Generate conflicts by constraint type
            detailed_conflicts.extend(self._detect_actor_conflicts(temp_ga, sequence, day_assignments, schedule))
            detailed_conflicts.extend(self._detect_director_conflicts(temp_ga, sequence, day_assignments, schedule))
            detailed_conflicts.extend(self._detect_location_conflicts(temp_ga, sequence, day_assignments, schedule))
            
            print(f"DEBUG: Generated {len(detailed_conflicts)} detailed conflict reports")
            
            # NEW: Generate constraint-level summary
            conflicts_summary = self._build_conflicts_summary(detailed_conflicts, temp_ga, sequence, day_assignments)
            
        except Exception as e:
            print(f"ERROR: Conflicts report generation failed: {e}")
            import traceback
            traceback.print_exc()
            detailed_conflicts = []
            conflicts_summary = self._empty_conflicts_summary()
        
        return {
            'detailed_conflicts': detailed_conflicts,
            'conflicts_summary': conflicts_summary
        }

    def _build_conflicts_summary(self, detailed_conflicts: List[Dict], ga_instance, 
                           sequence: List[int], day_assignments: List[int]) -> Dict:
        """NEW: Build constraint-level conflicts summary for UI grouping"""
        
        # Group detailed conflicts by constraint type
        conflicts_by_type = defaultdict(list)
        for conflict in detailed_conflicts:
            conflicts_by_type[conflict['type']].append(conflict)
        
        # Count unique constraint violations (should match hard_conflicts metric)
        unique_constraint_violations = self._count_unique_constraint_violations(
            ga_instance, sequence, day_assignments, detailed_conflicts)
        
        # Build summary by constraint category
        actor_violations = self._summarize_actor_violations(conflicts_by_type, ga_instance)
        director_violations = self._summarize_director_violations(conflicts_by_type, ga_instance)
        location_violations = self._summarize_location_violations(conflicts_by_type, ga_instance)
        
        # Calculate totals
        total_constraint_violations = (actor_violations['constraint_count'] + 
                                    director_violations['constraint_count'] + 
                                    location_violations['constraint_count'])
        
        conflicts_summary = {
            'total_detailed_reports': len(detailed_conflicts),
            'total_constraint_violations': total_constraint_violations,
            'constraint_categories': {
                'actor_constraints': actor_violations,
                'director_constraints': director_violations, 
                'location_constraints': location_violations
            },
            'conflicts_by_type': {
                conflict_type: {
                    'count': len(conflicts),
                    'severity_breakdown': self._get_severity_breakdown(conflicts)
                }
                for conflict_type, conflicts in conflicts_by_type.items()
            },
            'most_affected_scenes': self._get_most_affected_scenes(detailed_conflicts),
            'most_affected_days': self._get_most_affected_days(detailed_conflicts)
        }
        
        print(f"DEBUG: Conflicts summary - {total_constraint_violations} constraint violations, "
            f"{len(detailed_conflicts)} detailed reports")
        
        return conflicts_summary

    def _count_unique_constraint_violations(self, ga_instance, sequence: List[int], 
                                      day_assignments: List[int], detailed_conflicts: List[Dict]) -> Dict:
        """Count unique constraint violations (should match hard_conflicts metric)"""
        
        # Use the same violation counting logic as GA fitness function
        actor_violations = ga_instance._check_complete_actor_violations(sequence, day_assignments)
        director_violations = ga_instance._check_complete_director_violations(sequence, day_assignments)
        
        # TODO: Add other constraint types when implemented
        location_violations = 0
        equipment_violations = 0
        production_violations = 0
        
        return {
            'actor': actor_violations,
            'director': director_violations,
            'location': location_violations,
            'equipment': equipment_violations,
            'production': production_violations,
        }    
      
    def _summarize_actor_violations(self, conflicts_by_type: Dict, ga_instance) -> Dict:
        """Summarize actor constraint violations"""
        
        actor_conflict_types = ['actor_unavailable_date', 'actor_available_week', 'actor_required_days']
        actor_conflicts = []
        for conflict_type in actor_conflict_types:
            actor_conflicts.extend(conflicts_by_type.get(conflict_type, []))
        
        # Count unique actors with violations
        affected_actors = set()
        for conflict in actor_conflicts:
            if 'actor_name' in conflict:
                affected_actors.add(conflict['actor_name'])
        
        return {
            'constraint_count': len(ga_instance.actor_unavailable_dates) + len(ga_instance.actor_available_weeks) + len(ga_instance.actor_required_days),
            'violations_count': len([c for c in actor_conflicts if c.get('severity') == 'Hard']),
            'detailed_reports_count': len(actor_conflicts),
            'affected_actors_count': len(affected_actors),
            'affected_actors': list(affected_actors),
            'violation_types': {
                'unavailable_dates': len(conflicts_by_type.get('actor_unavailable_date', [])),
                'available_weeks': len(conflicts_by_type.get('actor_available_week', [])),
                'required_days': len(conflicts_by_type.get('actor_required_days', []))
            }
        }

    def _summarize_director_violations(self, conflicts_by_type: Dict, ga_instance) -> Dict:
        """Summarize director constraint violations"""
        
        director_conflict_types = ['director_shoot_first_violation', 'director_shoot_last_violation', 
                                'director_sequence_violation', 'director_same_day_violation']
        director_conflicts = []
        for conflict_type in director_conflict_types:
            director_conflicts.extend(conflicts_by_type.get(conflict_type, []))
        
        return {
            'constraint_count': (len(ga_instance.director_shoot_first) + len(ga_instance.director_shoot_last) + 
                            len(ga_instance.director_sequence_rules) + len(ga_instance.director_same_day_groups)),
            'violations_count': len([c for c in director_conflicts if c.get('severity') == 'Hard']),
            'detailed_reports_count': len(director_conflicts),
            'violation_types': {
                'shoot_first': len(conflicts_by_type.get('director_shoot_first_violation', [])),
                'shoot_last': len(conflicts_by_type.get('director_shoot_last_violation', [])),
                'sequence_rules': len(conflicts_by_type.get('director_sequence_violation', [])),
                'same_day_groups': len(conflicts_by_type.get('director_same_day_violation', []))
            }
        }

    def _summarize_location_violations(self, conflicts_by_type: Dict, ga_instance) -> Dict:
        """Summarize location constraint violations (placeholder)"""
        
        # TODO: Implement when location constraints are added
        return {
            'constraint_count': 0,
            'violations_count': 0,
            'detailed_reports_count': 0,
            'violation_types': {}
        }

    def _get_severity_breakdown(self, conflicts: List[Dict]) -> Dict:
        """Get breakdown of conflicts by severity"""
        severity_count = defaultdict(int)
        for conflict in conflicts:
            severity_count[conflict.get('severity', 'Unknown')] += 1
        return dict(severity_count)

    def _get_most_affected_scenes(self, detailed_conflicts: List[Dict], limit: int = 5) -> List[Dict]:
        """Get scenes most affected by conflicts"""
        scene_conflict_count = defaultdict(int)
        scene_conflict_types = defaultdict(set)
        
        for conflict in detailed_conflicts:
            for scene in conflict.get('affected_scenes', []):
                scene_conflict_count[scene] += 1
                scene_conflict_types[scene].add(conflict['type'])
        
        # Sort by conflict count and return top scenes
        top_scenes = sorted(scene_conflict_count.items(), key=lambda x: x[1], reverse=True)[:limit]
        
        return [
            {
                'scene_number': scene,
                'conflict_count': count,
                'conflict_types': list(scene_conflict_types[scene])
            }
            for scene, count in top_scenes
        ]

    def _get_most_affected_days(self, detailed_conflicts: List[Dict], limit: int = 5) -> List[Dict]:
        """Get days most affected by conflicts"""
        day_conflict_count = defaultdict(int)
        day_conflict_types = defaultdict(set)
        
        for conflict in detailed_conflicts:
            for day in conflict.get('affected_days', []):
                day_conflict_count[day] += 1
                day_conflict_types[day].add(conflict['type'])
        
        # Sort by conflict count and return top days
        top_days = sorted(day_conflict_count.items(), key=lambda x: x[1], reverse=True)[:limit]
        
        return [
            {
                'date': day,
                'conflict_count': count,
                'conflict_types': list(day_conflict_types[day])
            }
            for day, count in top_days
        ]

    def _empty_conflicts_summary(self) -> Dict:
        """Return empty conflicts summary structure"""
        return {
            'total_detailed_reports': 0,
            'total_constraint_violations': 0,
            'constraint_categories': {
                'actor_constraints': {'constraint_count': 0, 'violations_count': 0, 'detailed_reports_count': 0},
                'director_constraints': {'constraint_count': 0, 'violations_count': 0, 'detailed_reports_count': 0},
                'location_constraints': {'constraint_count': 0, 'violations_count': 0, 'detailed_reports_count': 0}
            },
            'conflicts_by_type': {},
            'most_affected_scenes': [],
            'most_affected_days': []
        }

    def _detect_actor_conflicts(self, ga_instance, sequence: List[int], day_assignments: List[int], 
                          schedule: List[Dict]) -> List[Dict]:
        """Detect and report actor constraint violations"""
        conflicts = []
        
        try:
            # Build scene-to-day mapping
            scene_to_day = ga_instance._build_scene_to_day_mapping(sequence, day_assignments)
            
            # Check actor unavailable dates
            conflicts.extend(self._check_actor_unavailable_date_conflicts(
                ga_instance, sequence, day_assignments, scene_to_day, schedule))
            
            # Check actor available weeks  
            conflicts.extend(self._check_actor_available_week_conflicts(
                ga_instance, sequence, day_assignments, scene_to_day, schedule))
            
            # Check actor required days
            conflicts.extend(self._check_actor_required_day_conflicts(
                ga_instance, sequence, day_assignments, scene_to_day, schedule))
            
        except Exception as e:
            print(f"ERROR: Actor conflict detection failed: {e}")
        
        return conflicts

    def _check_actor_unavailable_date_conflicts(self, ga_instance, sequence: List[int], 
                                          day_assignments: List[int], scene_to_day: Dict[str, int],
                                          schedule: List[Dict]) -> List[Dict]:
        """Check for actor unavailable date conflicts with detailed reporting"""
        conflicts = []
        
        for i, cluster_idx in enumerate(sequence):
            cluster = ga_instance.cluster_manager.clusters[cluster_idx]
            start_day = day_assignments[i]
            
            for day_offset in range(cluster.estimated_days):
                shooting_day_idx = start_day + day_offset
                if shooting_day_idx >= len(ga_instance.calendar.shooting_days):
                    continue
                
                shooting_date = ga_instance.calendar.shooting_days[shooting_day_idx]
                
                # Check each character in this cluster
                for character in cluster.required_actors:
                    actor = ga_instance._get_actor_for_character(character)
                    
                    if actor and actor in ga_instance.actor_unavailable_dates:
                        for unavailable_date_str in ga_instance.actor_unavailable_dates[actor]:
                            try:
                                unavailable_date = datetime.strptime(unavailable_date_str, "%Y-%m-%d").date()
                                if shooting_date == unavailable_date:
                                    # Find affected scenes on this day
                                    affected_scenes = []
                                    if shooting_day_idx < len(schedule):
                                        day_schedule = schedule[shooting_day_idx]
                                        affected_scenes = [scene['Scene_Number'] for scene in day_schedule.get('scenes', [])]
                                    
                                    conflicts.append({
                                        'type': 'actor_unavailable_date',
                                        'severity': 'Hard',
                                        'description': f"Actor '{actor}' (character '{character}') scheduled on unavailable date {shooting_date}",
                                        'affected_scenes': affected_scenes,
                                        'affected_days': [shooting_date.strftime("%Y-%m-%d")],
                                        'actor_name': actor,
                                        'character_name': character,
                                        'conflict_date': shooting_date.strftime("%Y-%m-%d")
                                    })
                            except:
                                pass
        
        return conflicts

    def _check_actor_available_week_conflicts(self, ga_instance, sequence: List[int], 
                                        day_assignments: List[int], scene_to_day: Dict[str, int],
                                        schedule: List[Dict]) -> List[Dict]:
        """Check for actor available week conflicts"""
        conflicts = []
        
        for i, cluster_idx in enumerate(sequence):
            cluster = ga_instance.cluster_manager.clusters[cluster_idx]
            start_day = day_assignments[i]
            
            for character in cluster.required_actors:
                actor = ga_instance._get_actor_for_character(character)
                
                if actor and actor in ga_instance.actor_available_weeks:
                    available_weeks = ga_instance.actor_available_weeks[actor]
                    
                    for day_offset in range(cluster.estimated_days):
                        day_idx = start_day + day_offset
                        if day_idx < len(ga_instance.calendar.shooting_days):
                            day_week = ga_instance._get_shooting_week_from_day(day_idx)
                            shooting_date = ga_instance.calendar.shooting_days[day_idx]
                            
                            if day_week not in available_weeks:
                                # Find affected scenes
                                affected_scenes = []
                                if day_idx < len(schedule):
                                    day_schedule = schedule[day_idx]
                                    affected_scenes = [scene['Scene_Number'] for scene in day_schedule.get('scenes', [])]
                                
                                conflicts.append({
                                    'type': 'actor_available_week',
                                    'severity': 'Hard',
                                    'description': f"Actor '{actor}' (character '{character}') scheduled in week {day_week}, only available weeks {available_weeks}",
                                    'affected_scenes': affected_scenes,
                                    'affected_days': [shooting_date.strftime("%Y-%m-%d")],
                                    'actor_name': actor,
                                    'character_name': character,
                                    'scheduled_week': day_week,
                                    'available_weeks': available_weeks
                                })
                                break
        
        return conflicts


    def _check_actor_required_day_conflicts(self, ga_instance, sequence: List[int], 
                                      day_assignments: List[int], scene_to_day: Dict[str, int],
                                      schedule: List[Dict]) -> List[Dict]:
        """Check for actor required days conflicts"""
        conflicts = []
        
        # Count actual scheduled days per character
        character_scheduled_days = defaultdict(int)
        character_scheduled_dates = defaultdict(list)
        
        for i, cluster_idx in enumerate(sequence):
            cluster = ga_instance.cluster_manager.clusters[cluster_idx]
            start_day = day_assignments[i]
            
            for character in cluster.required_actors:
                for day_offset in range(cluster.estimated_days):
                    day_idx = start_day + day_offset
                    if day_idx < len(ga_instance.calendar.shooting_days):
                        character_scheduled_days[character] += 1
                        shooting_date = ga_instance.calendar.shooting_days[day_idx]
                        character_scheduled_dates[character].append(shooting_date.strftime("%Y-%m-%d"))
        
        # Check against required days for each actor
        for actor, required_days in ga_instance.actor_required_days.items():
            characters_for_actor = [char for char, mapped_actor in ga_instance.cast_mapping.items() 
                                if mapped_actor == actor]
            
            total_scheduled_days = 0
            all_scheduled_dates = []
            affected_scenes = []
            
            for character in characters_for_actor:
                total_scheduled_days += character_scheduled_days.get(character, 0)
                all_scheduled_dates.extend(character_scheduled_dates.get(character, []))
                
                # Find scenes for this character
                for scene_num, day_idx in scene_to_day.items():
                    if day_idx < len(schedule):
                        day_schedule = schedule[day_idx]
                        for scene in day_schedule.get('scenes', []):
                            if scene['Scene_Number'] == scene_num:
                                cast = scene.get('Cast', [])
                                if character in cast:
                                    affected_scenes.append(scene_num)
            
            if total_scheduled_days != required_days:
                conflicts.append({
                    'type': 'actor_required_days',
                    'severity': 'Hard',
                    'description': f"Actor '{actor}' scheduled for {total_scheduled_days} days, requires {required_days} days",
                    'affected_scenes': list(set(affected_scenes)),
                    'affected_days': list(set(all_scheduled_dates)),
                    'actor_name': actor,
                    'characters_played': characters_for_actor,
                    'scheduled_days': total_scheduled_days,
                    'required_days': required_days,
                    'days_difference': total_scheduled_days - required_days
                })
        
        return conflicts    

    def _detect_director_conflicts(self, ga_instance, sequence: List[int], day_assignments: List[int],
                             schedule: List[Dict]) -> List[Dict]:
        """Detect and report director constraint violations"""
        conflicts = []
        
        try:
            # Build scene-to-day mapping
            scene_to_day = ga_instance._build_scene_to_day_mapping(sequence, day_assignments)
            
            # Check shoot first violations
            conflicts.extend(self._check_director_shoot_first_conflicts(
                ga_instance, scene_to_day, schedule))
            
            # Check shoot last violations
            conflicts.extend(self._check_director_shoot_last_conflicts(
                ga_instance, scene_to_day, schedule))
            
            # Check sequence rule violations
            conflicts.extend(self._check_director_sequence_conflicts(
                ga_instance, scene_to_day, schedule))
            
            # Check same day group violations
            conflicts.extend(self._check_director_same_day_conflicts(
                ga_instance, scene_to_day, schedule))
            
        except Exception as e:
            print(f"ERROR: Director conflict detection failed: {e}")
        
        return conflicts

    def _check_director_shoot_first_conflicts(self, ga_instance, scene_to_day: Dict[str, int],
                                        schedule: List[Dict]) -> List[Dict]:
        """Check director shoot first violations"""
        conflicts = []
        
        if not ga_instance.director_shoot_first:
            return conflicts
        
        # Find earliest day among mandated scenes
        earliest_mandated_day = float('inf')
        for scene_num in ga_instance.director_shoot_first:
            if scene_num in scene_to_day:
                earliest_mandated_day = min(earliest_mandated_day, scene_to_day[scene_num])
        
        # Find scenes scheduled before earliest mandated scene
        if earliest_mandated_day != float('inf'):
            for scene_num, day in scene_to_day.items():
                if scene_num not in ga_instance.director_shoot_first and day < earliest_mandated_day:
                    shooting_date = ga_instance.calendar.shooting_days[day] if day < len(ga_instance.calendar.shooting_days) else None
                    
                    conflicts.append({
                        'type': 'director_shoot_first_violation',
                        'severity': 'Hard',
                        'description': f"Scene {scene_num} scheduled before 'shoot first' scenes {ga_instance.director_shoot_first}",
                        'affected_scenes': [scene_num] + ga_instance.director_shoot_first,
                        'affected_days': [shooting_date.strftime("%Y-%m-%d")] if shooting_date else [],
                        'violating_scene': scene_num,
                        'mandated_first_scenes': ga_instance.director_shoot_first
                    })
        
        return conflicts    
    def _check_director_shoot_last_conflicts(self, ga_instance, scene_to_day: Dict[str, int],
                                       schedule: List[Dict]) -> List[Dict]:
        """Check director shoot last violations"""
        conflicts = []
        
        if not ga_instance.director_shoot_last:
            return conflicts
        
        # Find latest day among mandated scenes
        latest_mandated_day = -1
        for scene_num in ga_instance.director_shoot_last:
            if scene_num in scene_to_day:
                latest_mandated_day = max(latest_mandated_day, scene_to_day[scene_num])
        
        # Find scenes scheduled after latest mandated scene
        if latest_mandated_day >= 0:
            for scene_num, day in scene_to_day.items():
                if scene_num not in ga_instance.director_shoot_last and day > latest_mandated_day:
                    shooting_date = ga_instance.calendar.shooting_days[day] if day < len(ga_instance.calendar.shooting_days) else None
                    
                    conflicts.append({
                        'type': 'director_shoot_last_violation',
                        'severity': 'Hard',
                        'description': f"Scene {scene_num} scheduled after 'shoot last' scenes {ga_instance.director_shoot_last}",
                        'affected_scenes': [scene_num] + ga_instance.director_shoot_last,
                        'affected_days': [shooting_date.strftime("%Y-%m-%d")] if shooting_date else [],
                        'violating_scene': scene_num,
                        'mandated_last_scenes': ga_instance.director_shoot_last
                    })
        
        return conflicts

    def _check_director_sequence_conflicts(self, ga_instance, scene_to_day: Dict[str, int],
                                     schedule: List[Dict]) -> List[Dict]:
        """Check director sequence rule violations"""
        conflicts = []
        
        for rule in ga_instance.director_sequence_rules:
            rule_type = rule.get('type')
            
            if rule_type == 'before':
                first_scene = rule['first_scene']
                second_scene = rule['second_scene']
                
                if (first_scene in scene_to_day and second_scene in scene_to_day):
                    if scene_to_day[first_scene] >= scene_to_day[second_scene]:
                        first_date = ga_instance.calendar.shooting_days[scene_to_day[first_scene]] if scene_to_day[first_scene] < len(ga_instance.calendar.shooting_days) else None
                        second_date = ga_instance.calendar.shooting_days[scene_to_day[second_scene]] if scene_to_day[second_scene] < len(ga_instance.calendar.shooting_days) else None
                        
                        conflicts.append({
                            'type': 'director_sequence_violation',
                            'severity': 'Hard',
                            'description': f"Scene {first_scene} must be shot before Scene {second_scene}, but scheduled after",
                            'affected_scenes': [first_scene, second_scene],
                            'affected_days': [d.strftime("%Y-%m-%d") for d in [first_date, second_date] if d],
                            'sequence_rule': rule,
                            'first_scene': first_scene,
                            'second_scene': second_scene,
                            'rule_type': 'before'
                        })
            
            elif rule_type == 'after':
                first_scene = rule['first_scene']
                second_scene = rule['second_scene']
                
                if (first_scene in scene_to_day and second_scene in scene_to_day):
                    if scene_to_day[first_scene] <= scene_to_day[second_scene]:
                        first_date = ga_instance.calendar.shooting_days[scene_to_day[first_scene]] if scene_to_day[first_scene] < len(ga_instance.calendar.shooting_days) else None
                        second_date = ga_instance.calendar.shooting_days[scene_to_day[second_scene]] if scene_to_day[second_scene] < len(ga_instance.calendar.shooting_days) else None
                        
                        conflicts.append({
                            'type': 'director_sequence_violation',
                            'severity': 'Hard',
                            'description': f"Scene {first_scene} must be shot after Scene {second_scene}, but scheduled before",
                            'affected_scenes': [first_scene, second_scene],
                            'affected_days': [d.strftime("%Y-%m-%d") for d in [first_date, second_date] if d],
                            'sequence_rule': rule,
                            'first_scene': first_scene,
                            'second_scene': second_scene,
                            'rule_type': 'after'
                        })
        
        return conflicts

    def _check_director_same_day_conflicts(self, ga_instance, scene_to_day: Dict[str, int],
                                     schedule: List[Dict]) -> List[Dict]:
        """Check director same day grouping violations"""
        conflicts = []
        
        for group in ga_instance.director_same_day_groups:
            scenes = group['scenes']
            
            # Get days for all scenes in group
            group_days = []
            scene_dates = []
            for scene_num in scenes:
                if scene_num in scene_to_day:
                    day_idx = scene_to_day[scene_num]
                    group_days.append(day_idx)
                    if day_idx < len(ga_instance.calendar.shooting_days):
                        scene_dates.append(ga_instance.calendar.shooting_days[day_idx].strftime("%Y-%m-%d"))
            
            # Check if scenes are spread across multiple days
            if len(set(group_days)) > 1:
                conflicts.append({
                    'type': 'director_same_day_violation',
                    'severity': 'Hard',
                    'description': f"Scenes {scenes} must be shot on same day but scheduled across {len(set(group_days))} different days",
                    'affected_scenes': scenes,
                    'affected_days': list(set(scene_dates)),
                    'same_day_group': group,
                    'days_spread': len(set(group_days)),
                    'reasoning': group.get('reasoning', '')
                })
        
        return conflicts    

    def _detect_location_conflicts(self, ga_instance, sequence: List[int], day_assignments: List[int],
                             schedule: List[Dict]) -> List[Dict]:
        """Detect location-related conflicts (placeholder for future location constraints)"""
        conflicts = []
        
        # TODO: Implement location availability window violations
        # TODO: Implement equipment rental period violations
        # TODO: Implement production rule violations
        
        return conflicts

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
        """Split a complex scene into multiple parts - MINIMAL LOGGING"""
        
        # Calculate how many parts we need
        total_parts = int((total_hours + max_daily_hours - 0.1) / max_daily_hours)
        
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
            
            # Create scene part
            scene_part = scene.copy()
            
            # Add hybrid fields
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
        
        # REDUCED: Only one summary log per scene split (not per part)
        print(f"DEBUG: Split Scene {scene['Scene_Number']} ({total_hours:.1f}h) into {total_parts} parts")
        
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
                'hard_conflicts': 0,
                'soft_conflicts': 0
            }
        
        # Count unique scenes (prevent duplication counting)
        unique_scenes = set()
        for day in schedule:
            for scene in day['scenes']:
                unique_scenes.add(scene['Scene_Number'])
        
        total_moves = sum(day.get('location_moves', 0) for day in schedule)
        n_unique_locations = len(self.cluster_manager.clusters)
        
        # Calculate geographic location moves (crew moves between locations)
        total_geographic_moves = self._calculate_geographic_moves(schedule)
        
        # Correct theoretical minimum moves
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
        print(f"DEBUG: Starting metrics hard conflicts calculation...")
        hard_conflicts = self._calculate_real_hard_conflicts_from_schedule(schedule)
        print(f"DEBUG: Metrics calculated {hard_conflicts} hard conflicts")
        
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
            'soft_conflicts': 0
        }

    def _calculate_real_hard_conflicts_from_schedule(self, schedule: List[Dict]) -> int:
        """Calculate hard conflicts by converting schedule back to GA format - MINIMAL LOGGING"""
        try:
            # Convert schedule back to GA individual format
            sequence, day_assignments = self._schedule_to_ga_format(schedule)
            
            if not sequence or not day_assignments:
                return 0
            
            # Create a temporary GA instance to use violation detection methods
            temp_ga = LocationFirstGA(self.cluster_manager, self.constraints, self.calendar, 
                                    self.params, self.cast_mapping)
            
            # Use the same violation detection as fitness function
            actor_violations = temp_ga._check_complete_actor_violations(sequence, day_assignments)
            
            total_hard_conflicts = actor_violations
            
            # REMOVED: All debug prints
            return total_hard_conflicts
            
        except Exception as e:
            print(f"ERROR: Metrics calculation failed: {e}")
            return 0


    def _schedule_to_ga_format(self, schedule: List[Dict]) -> Tuple[List[int], List[int]]:
        """Convert final schedule back to GA format for violation checking"""
        try:
            sequence = []
            day_assignments = []
            
            print(f"DEBUG: Converting schedule with {len(schedule)} days")
            
            # Map locations to cluster indices
            location_to_cluster = {}
            for i, cluster in enumerate(self.cluster_manager.clusters):
                location_to_cluster[cluster.location] = i
            
            print(f"DEBUG: Location to cluster mapping: {len(location_to_cluster)} locations")
            
            seen_clusters = set()
            
            for day_idx, day in enumerate(schedule):
                location = day.get('location', '')
                cluster_idx = location_to_cluster.get(location)
                
                if cluster_idx is not None and cluster_idx not in seen_clusters:
                    sequence.append(cluster_idx)
                    day_assignments.append(day_idx)
                    seen_clusters.add(cluster_idx)
                    print(f"DEBUG: Added cluster {cluster_idx} starting day {day_idx} for location '{location}'")
            
            print(f"DEBUG: Final sequence: {sequence}, assignments: {day_assignments}")
            return sequence, day_assignments
            
        except Exception as e:
            print(f"DEBUG: Exception in _schedule_to_ga_format: {e}")
            import traceback
            traceback.print_exc()
            return [], []

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
        """Extract scene time estimates from operational data - NO VERBOSE LOGGING"""
        scene_estimates = {}
        
        try:
            if 'operational_data' in self.constraints_raw:
                operational_data = self.constraints_raw['operational_data']
                
                if ('time_estimates' in operational_data and 
                    'scene_estimates' in operational_data['time_estimates']):
                    
                    scene_estimates_data = operational_data['time_estimates']['scene_estimates']
                    
                    if isinstance(scene_estimates_data, list):
                        for scene_est in scene_estimates_data:
                            if isinstance(scene_est, dict):
                                scene_number = None
                                estimated_hours = 1.5
                                
                                # Try multiple key variations for scene number
                                for key in scene_est.keys():
                                    if 'Scene_Number' in key:
                                        scene_number = scene_est[key]
                                        break
                                
                                # Try multiple key variations for estimated hours
                                for key in scene_est.keys():
                                    if 'Estimated_Time_Hours' in key:
                                        estimated_hours = scene_est[key]
                                        break
                                
                                if scene_number:
                                    try:
                                        hours_float = float(estimated_hours) if estimated_hours else 1.5
                                        scene_estimates[str(scene_number)] = hours_float
                                        # REMOVED: Debug print for every scene
                                    except (ValueError, TypeError):
                                        scene_estimates[str(scene_number)] = 1.5
            
            # SINGLE summary log only
            print(f"DEBUG: Loaded {len(scene_estimates)} scene time estimates")
            
        except Exception as e:
            print(f"ERROR: Scene estimates extraction failed: {e}")
    
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
            conflicts_summary=result['conflicts_summary'],
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
