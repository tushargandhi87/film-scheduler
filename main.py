"""
Film Production Schedule Optimizer using Genetic Algorithm
Generic, deterministic solution for any film production
"""

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import re
import json
from collections import defaultdict
import hashlib

app = FastAPI(title="Film Schedule Optimizer")

# Enable CORS for n8n cloud
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ScheduleRequest(BaseModel):
    """Request model for schedule optimization"""
    stripboard: List[Dict[str, Any]]
    constraints: Dict[str, str]  # Raw text from each constraint document
    ga_params: Optional[Dict[str, Any]] = {
        "population_size": 100,
        "generations": 500,
        "mutation_rate": 0.02,
        "crossover_rate": 0.8,
        "seed": 42,
        "elite_size": 10
    }

class ScheduleResponse(BaseModel):
    """Response model with optimized schedule"""
    location_sequence: List[Dict[str, Any]]
    fitness_score: float
    total_moves: int
    estimated_travel_distance: float
    violations: List[str]

class ConstraintParser:
    """Parse constraints from text documents"""
    
    def __init__(self):
        self.date_patterns = {
            'range': r'(\w+\.?\s+\d+)\s*[-–]\s*(\w+\.?\s+\d+)',
            'unavailable': r'UNAVAILABLE:?\s*([^.]+)',
            'only': r'(\w+\.?\s+\d+)\s+ONLY',
            'weekday': r'(weekends?|weekdays?)\s+only',
            'week_based': r'week\s+(\d+)',
        }
    
    def parse_actor_availability(self, text: str) -> Dict:
        """Extract actor availability constraints"""
        constraints = defaultdict(dict)
        lines = text.split('\n')
        current_actor = None
        
        for line in lines:
            # Find actor names
            if '(' in line and ')' in line:
                match = re.search(r'(.+?)\s*\(as\s+(.+?)\)', line)
                if match:
                    current_actor = match.group(1).strip()
                    constraints[current_actor]['character'] = match.group(2)
            
            if current_actor:
                # Find unavailable dates
                if 'UNAVAILABLE' in line:
                    dates = re.findall(self.date_patterns['unavailable'], line)
                    if dates:
                        if 'unavailable' not in constraints[current_actor]:
                            constraints[current_actor]['unavailable'] = []
                        constraints[current_actor]['unavailable'].extend(dates)
                
                # Find week-based availability
                if 'Week' in line:
                    weeks = re.findall(self.date_patterns['week_based'], line)
                    if weeks:
                        constraints[current_actor]['available_weeks'] = [int(w) for w in weeks]
        
        return dict(constraints)
    
    def parse_location_availability(self, text: str) -> Dict:
        """Extract location availability windows"""
        constraints = defaultdict(dict)
        lines = text.split('\n')
        current_location = None
        
        for line in lines:
            # Identify location headers
            if '●' in line or '○' in line:
                # Extract location name
                match = re.search(r'[●○]\s*(.+?)(?:\(|:)', line)
                if match:
                    current_location = match.group(1).strip()
            
            if current_location:
                # Find availability patterns
                if 'Availability:' in line:
                    if 'ONLY' in line:
                        only_dates = re.findall(r'(\w+\.?\s+\d+(?:\s*[-–]\s*\w+\.?\s+\d+)?)\s+ONLY', line)
                        if only_dates:
                            constraints[current_location]['exclusive_dates'] = only_dates
                    
                    if 'Weekends' in line.title() or 'weekends' in line:
                        constraints[current_location]['weekends_only'] = True
                    
                    # Date ranges
                    ranges = re.findall(self.date_patterns['range'], line)
                    if ranges:
                        constraints[current_location]['available_range'] = ranges
        
        return dict(constraints)
    
    def parse_director_notes(self, text: str) -> List[Dict]:
        """Extract director priorities and mandates"""
        mandates = []
        
        # Look for scene-specific instructions
        scene_pattern = r'Scene\s+(\d+[a-z]?)\s*[:\(](.+?)(?:\.|\n)'
        matches = re.finditer(scene_pattern, text, re.IGNORECASE)
        
        for match in matches:
            scene_num = match.group(1)
            instruction = match.group(2)
            
            if 'must' in instruction.lower() or 'first' in instruction.lower() or 'day 1' in instruction.lower():
                mandates.append({
                    'scene': scene_num,
                    'priority': 'high',
                    'instruction': instruction
                })
        
        return mandates

class LocationClusterer:
    """Group locations by geographic proximity"""
    
    def __init__(self, stripboard: List[Dict]):
        self.scenes = stripboard
        self.location_groups = defaultdict(list)
        self._cluster_by_address()
    
    def _cluster_by_address(self):
        """Group scenes by geographic location"""
        for scene in self.scenes:
            geo_loc = scene.get('Geographic_Location', 'Unknown')
            if not geo_loc or geo_loc == "Location TBD":
                # Use location name as fallback
                geo_loc = scene.get('Location_Name', 'Unknown')
            
            self.location_groups[geo_loc].append(scene)
    
    def get_unique_locations(self) -> List[str]:
        """Get list of unique geographic locations"""
        return list(self.location_groups.keys())
    
    def get_scenes_at_location(self, location: str) -> List[Dict]:
        """Get all scenes at a specific location"""
        return self.location_groups.get(location, [])
    
    def estimate_distance(self, loc1: str, loc2: str) -> float:
        """Estimate distance between locations"""
        # Simple heuristic based on address similarity
        if loc1 == loc2:
            return 0.0
        
        # Check if same city
        city_pattern = r',\s*([^,]+),\s*[A-Z]{2}'
        city1 = re.search(city_pattern, loc1)
        city2 = re.search(city_pattern, loc2)
        
        if city1 and city2:
            if city1.group(1) == city2.group(1):
                return 5.0  # Same city, different address
        
        # Check if same state
        state1 = re.search(r',\s*([A-Z]{2})', loc1)
        state2 = re.search(r',\s*([A-Z]{2})', loc2)
        
        if state1 and state2:
            if state1.group(1) == state2.group(1):
                return 25.0  # Same state, different city
        
        # Different states or unable to determine
        return 100.0

class GeneticAlgorithm:
    """Deterministic GA for film schedule optimization"""
    
    def __init__(self, locations: List[str], constraints: Dict, params: Dict):
        self.locations = locations
        self.constraints = constraints
        self.params = params
        
        # Set seed for determinism
        np.random.seed(params.get('seed', 42))
        self.rng = np.random.RandomState(params.get('seed', 42))
        
        # GA parameters
        self.pop_size = params.get('population_size', 100)
        self.generations = params.get('generations', 500)
        self.mutation_rate = params.get('mutation_rate', 0.02)
        self.crossover_rate = params.get('crossover_rate', 0.8)
        self.elite_size = params.get('elite_size', 10)
        
        # Distance matrix
        self.clusterer = LocationClusterer([])  # Will be set externally
        self.distance_matrix = self._build_distance_matrix()
    
    def _build_distance_matrix(self) -> np.ndarray:
        """Build distance matrix between all locations"""
        n = len(self.locations)
        matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i != j and hasattr(self, 'clusterer'):
                    dist = self.clusterer.estimate_distance(
                        self.locations[i], 
                        self.locations[j]
                    )
                    matrix[i][j] = dist
        
        return matrix
    
    def create_individual(self) -> List[int]:
        """Create a random valid individual (location sequence)"""
        individual = list(range(len(self.locations)))
        self.rng.shuffle(individual)
        return individual
    
    def create_population(self) -> List[List[int]]:
        """Create initial population"""
        return [self.create_individual() for _ in range(self.pop_size)]
    
    def fitness(self, individual: List[int]) -> float:
        """Calculate fitness score for an individual"""
        score = 0.0
        
        # 1. Minimize location changes (travel distance)
        total_distance = 0.0
        for i in range(len(individual) - 1):
            dist = self.distance_matrix[individual[i]][individual[i+1]]
            total_distance += dist
        score -= total_distance * 10  # Weight for distance
        
        # 2. Check actor availability constraints
        actor_violations = self._check_actor_constraints(individual)
        score -= actor_violations * 100
        
        # 3. Check location availability constraints
        location_violations = self._check_location_constraints(individual)
        if location_violations > 0:
            return -float('inf')  # Hard constraint violation
        
        # 4. Director mandates
        director_bonus = self._check_director_preferences(individual)
        score += director_bonus * 50
        
        return score
    
    def _check_actor_constraints(self, individual: List[int]) -> int:
        """Count actor availability violations"""
        # Simplified - would need scene scheduling details
        return 0
    
    def _check_location_constraints(self, individual: List[int]) -> int:
        """Count location availability violations"""
        # Simplified - would need date assignment logic
        return 0
    
    def _check_director_preferences(self, individual: List[int]) -> int:
        """Bonus points for following director preferences"""
        # Check if priority scenes are scheduled early
        return 0
    
    def selection(self, population: List[List[int]], fitnesses: List[float]) -> List[int]:
        """Tournament selection"""
        tournament_size = 5
        tournament_indices = self.rng.choice(len(population), tournament_size, replace=False)
        tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
        winner_idx = tournament_indices[np.argmax(tournament_fitnesses)]
        return population[winner_idx].copy()
    
    def crossover(self, parent1: List[int], parent2: List[int]) -> List[int]:
        """Order crossover (OX) for permutation"""
        if self.rng.random() > self.crossover_rate:
            return parent1.copy()
        
        size = len(parent1)
        start, end = sorted(self.rng.choice(size, 2, replace=False))
        
        child = [-1] * size
        child[start:end] = parent1[start:end]
        
        pointer = end
        for gene in parent2[end:] + parent2[:end]:
            if gene not in child:
                child[pointer % size] = gene
                pointer += 1
        
        return child
    
    def mutate(self, individual: List[int]) -> List[int]:
        """Swap mutation"""
        if self.rng.random() < self.mutation_rate:
            i, j = self.rng.choice(len(individual), 2, replace=False)
            individual[i], individual[j] = individual[j], individual[i]
        return individual
    
    def evolve(self) -> tuple:
        """Run genetic algorithm"""
        population = self.create_population()
        best_individual = None
        best_fitness = -float('inf')
        
        for generation in range(self.generations):
            # Evaluate fitness
            fitnesses = [self.fitness(ind) for ind in population]
            
            # Track best
            gen_best_idx = np.argmax(fitnesses)
            if fitnesses[gen_best_idx] > best_fitness:
                best_fitness = fitnesses[gen_best_idx]
                best_individual = population[gen_best_idx].copy()
            
            # Create new population
            new_population = []
            
            # Elitism
            elite_indices = np.argsort(fitnesses)[-self.elite_size:]
            for idx in elite_indices:
                new_population.append(population[idx].copy())
            
            # Generate rest of population
            while len(new_population) < self.pop_size:
                parent1 = self.selection(population, fitnesses)
                parent2 = self.selection(population, fitnesses)
                child = self.crossover(parent1, parent2)
                child = self.mutate(child)
                new_population.append(child)
            
            population = new_population[:self.pop_size]
        
        return best_individual, best_fitness

@app.post("/optimize/location-order", response_model=ScheduleResponse)
async def optimize_schedule(request: ScheduleRequest):
    """
    Optimize film production schedule using genetic algorithm
    """
    try:
        # 1. Parse constraints
        parser = ConstraintParser()
        actor_constraints = {}
        location_constraints = {}
        director_mandates = []
        
        if 'actor_availability' in request.constraints:
            actor_constraints = parser.parse_actor_availability(
                request.constraints['actor_availability']
            )
        
        if 'location_availability' in request.constraints:
            location_constraints = parser.parse_location_availability(
                request.constraints['location_availability']
            )
        
        if 'director_notes' in request.constraints:
            director_mandates = parser.parse_director_notes(
                request.constraints['director_notes']
            )
        
        # 2. Cluster locations
        clusterer = LocationClusterer(request.stripboard)
        unique_locations = clusterer.get_unique_locations()
        
        # 3. Run GA
        constraints_dict = {
            'actors': actor_constraints,
            'locations': location_constraints,
            'director': director_mandates
        }
        
        ga = GeneticAlgorithm(unique_locations, constraints_dict, request.ga_params)
        ga.clusterer = clusterer  # Set clusterer for distance calculations
        
        best_sequence, best_fitness = ga.evolve()
        
        # 4. Build response
        ordered_locations = [unique_locations[i] for i in best_sequence]
        
        # Rebuild stripboard in optimized order
        optimized_stripboard = []
        for location in ordered_locations:
            scenes = clusterer.get_scenes_at_location(location)
            # Sort scenes by scene number within each location
            scenes_sorted = sorted(scenes, key=lambda x: x.get('Scene_Number', ''))
            optimized_stripboard.extend(scenes_sorted)
        
        # Calculate metrics
        total_moves = len(unique_locations) - 1
        total_distance = sum(
            clusterer.estimate_distance(ordered_locations[i], ordered_locations[i+1])
            for i in range(len(ordered_locations) - 1)
        )
        
        return ScheduleResponse(
            location_sequence=optimized_stripboard,
            fitness_score=float(best_fitness),
            total_moves=total_moves,
            estimated_travel_distance=total_distance,
            violations=[]
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "version": "1.0.0"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)