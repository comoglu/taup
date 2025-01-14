#!/usr/bin/env python3

"""
Seismic Wave Analysis Tool
-------------------------
A tool for analyzing seismic wave propagation through the Earth using different velocity models.
"""

from obspy.taup import TauPyModel
import numpy as np
import matplotlib.pyplot as plt
import argparse
from typing import List, Dict, Optional, Set

class SeismicAnalyzer:
    def __init__(self):
        """Initialize SeismicAnalyzer with multiple velocity models."""
        self.models = {
            'iasp91': TauPyModel(model="iasp91"),
            'ak135': TauPyModel(model="ak135"),
            'prem': TauPyModel(model="prem")
        }
        
        # Define phase categories and recommended distances
        self.phase_categories = {
            'Regional (<15°)': {
                'P waves': ['Pg', 'Pn'],
                'S waves': ['Sg', 'Sn'],
                'Other': ['Lg']
            },
            'Teleseismic (>10°)': {
                'Direct waves': ['P', 'S'],
                'Depth phases': ['pP', 'sP', 'pS', 'sS'],
                'Surface reflections': ['PP', 'SS'],
                'Core reflections': ['PcP', 'ScS'],
                'Core phases': ['PKP', 'SKS', 'PKIKP']
            }
        }

    def calculate_travel_times(self, depth: float, distance: float, phases: List[str], 
                             model_name: str = 'ak135') -> None:
        """Calculate and display travel times for seismic phases."""
        model = self.models[model_name]
        arrivals = model.get_travel_times(depth, distance, phase_list=phases)
        
        if not arrivals:
            print(f"\nNo arrivals found for:")
            print(f"Depth: {depth} km")
            print(f"Distance: {distance}°")
            print(f"Phases: {', '.join(phases)}")
            return

        # Print header
        print(f"\nTravel Times Results ({model_name.upper()})")
        print(f"Source depth: {depth} km, Distance: {distance}°")
        print("\nPhase      Time(s)    Take-off    Description")
        print("-" * 55)

        # Group arrivals by phase name
        phase_groups = {}
        for arr in sorted(arrivals, key=lambda x: x.time):
            if arr.name not in phase_groups:
                phase_groups[arr.name] = []
            phase_groups[arr.name].append(arr)

        # Display results in a compact format
        for phase_name, phase_arrivals in phase_groups.items():
            if len(phase_arrivals) == 1:
                # Single arrival
                arr = phase_arrivals[0]
                print(f"{phase_name:<10} {arr.time:<10.2f} {arr.takeoff_angle:<10.1f} {self._get_path_description(arr.takeoff_angle)}")
            else:
                # Multiple arrivals
                print(f"{phase_name} (multiple paths):")
                for i, arr in enumerate(sorted(phase_arrivals, key=lambda x: x.time), 1):
                    print(f"  {i:<8} {arr.time:<10.2f} {arr.takeoff_angle:<10.1f} {self._get_path_description(arr.takeoff_angle)}")

    def plot_ray_paths(self, depth: float, distance: float, phases: List[str], 
                      model_name: str = 'ak135') -> None:
        """Plot ray paths for seismic phases."""
        model = self.models[model_name]
        plt.figure(figsize=(12, 8))
        
        self._plot_earth_structure()
        
        # Plot ray paths
        for phase in phases:
            try:
                arrivals = model.get_ray_paths(depth, distance, phase_list=[phase])
                for arr in arrivals:
                    plt.plot(arr.path['dist'], arr.path['depth'],
                           label=f'{phase} - {arr.time:.1f}s')
            except Exception as e:
                print(f"Could not plot {phase}: {str(e)}")
        
        plt.gca().invert_yaxis()
        plt.xlabel('Distance (degrees)')
        plt.ylabel('Depth (km)')
        plt.title(f'Seismic Ray Paths ({model_name.upper()})\n'
                 f'Source Depth = {depth} km, Distance = {distance}°')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def _plot_earth_structure(self):
        """Plot Earth's internal structure."""
        radius = 6371  # Earth's radius in km
        discontinuities = {
            'CMB': 2890,  # Core-mantle boundary
            '660': 660,   # 660 km discontinuity
            '410': 410    # 410 km discontinuity
        }
        
        theta = np.linspace(0, np.pi, 100)
        
        # Plot surface
        plt.plot(np.degrees(theta), radius - radius * np.cos(theta), 
                'k--', alpha=0.3, label='Surface')
        
        # Plot major discontinuities
        for name, depth in discontinuities.items():
            plt.plot(np.degrees(theta), radius - (radius - depth) * np.cos(theta),
                    ':', alpha=0.3, label=name)

    def _get_path_description(self, takeoff_angle: Optional[float]) -> str:
        """Get path description based on takeoff angle."""
        if takeoff_angle is None:
            return "Path info unavailable"
        if takeoff_angle < 45:
            return "Steep downgoing path"
        if takeoff_angle < 90:
            return "Shallow downgoing path"
        if takeoff_angle < 135:
            return "Shallow upgoing path"
        return "Steep upgoing path"
    def generate_time_table(self, depth: float, distances: List[float], phases: List[str],
                          models: Optional[List[str]] = None) -> None:
        """Generate a travel time table for multiple distances and models."""
        if models is None:
            models = ['ak135']
            
        for model_name in models:
            model = self.models[model_name]
            print(f"\nTravel Time Table ({model_name.upper()})")
            print("-" * (12 + 12 * len(phases)))
            
            # Print header
            header = "Distance(°) "
            header += " ".join(f"{phase:<11}" for phase in phases)
            print(header)
            print("-" * (12 + 12 * len(phases)))
            
            for dist in distances:
                arrivals = model.get_travel_times(depth, dist, phase_list=phases)
                times = {arr.name: arr.time for arr in arrivals}
                
                row = f"{dist:<10.1f} "
                for phase in phases:
                    time = times.get(phase, None)
                    row += f"{time:11.1f} " if time is not None else "    ---     "
                print(row)
            print()

def get_float_input(prompt: str, min_val: float, max_val: float) -> float:
    """Get validated float input from user."""
    while True:
        try:
            value = float(input(prompt))
            if min_val <= value <= max_val:
                return value
            print(f"Value must be between {min_val} and {max_val}")
        except ValueError:
            print("Please enter a valid number")

def display_phase_menu() -> List[str]:
    """Display interactive phase selection menu."""
    analyzer = SeismicAnalyzer()
    
    while True:
        print("\nPhase Selection Menu:")
        print("1. Select from suggested phases (recommended)")
        print("2. Enter phases manually")
        choice = input("Choose an option (1-2): ").strip()
        
        if choice == '1':
            distance = get_float_input("Enter distance (degrees) for phase suggestions: ", 0, 180)
            suggested = get_suggested_phases(distance)
            
            print("\nSuggested phases for this distance:")
            for i, phase in enumerate(suggested, 1):
                print(f"{i}. {phase}")
            
            selections = input("\nEnter numbers of phases to use (e.g., '1,2,3' or '1-3'): ").strip()
            try:
                if '-' in selections:
                    start, end = map(int, selections.split('-'))
                    selected_indices = range(start, end + 1)
                else:
                    selected_indices = map(int, selections.split(','))
                
                selected_phases = [suggested[i-1] for i in selected_indices]
                return selected_phases
            except (ValueError, IndexError):
                print("Invalid selection. Please try again.")
        
        elif choice == '2':
            print("\nAvailable phase categories:")
            for category, subcategories in analyzer.phase_categories.items():
                print(f"\n{category}:")
                for subcat, phases in subcategories.items():
                    print(f"  {subcat}: {', '.join(phases)}")
            
            phase_input = input("\nEnter phases separated by spaces or commas: ")
            if ',' in phase_input:
                phases = [p.strip() for p in phase_input.split(',')]
            else:
                phases = phase_input.split()
            return [p for p in phases if p]

def get_suggested_phases(distance: float) -> List[str]:
    """Get suggested phases based on distance."""
    suggested = []
    if distance <= 15:
        suggested.extend(['Pg', 'Pn', 'Sg', 'Sn'])
        if distance > 2:
            suggested.append('Lg')
    if distance > 10:
        suggested.extend(['P', 'S'])
    if distance > 20:
        suggested.extend(['PP', 'SS', 'PcP', 'ScS'])
    if distance > 100:
        suggested.extend(['PKP', 'SKS', 'PKIKP'])
    return suggested

def select_models() -> List[str]:
    """Select velocity models to use."""
    available_models = ['iasp91', 'ak135', 'prem']
    
    print("\nAvailable velocity models:")
    for i, model in enumerate(available_models, 1):
        print(f"{i}. {model.upper()}")
    
    while True:
        selection = input("Enter model numbers to compare (e.g., '1,2' or '1-3'): ").strip()
        try:
            if '-' in selection:
                start, end = map(int, selection.split('-'))
                indices = range(start-1, end)
            else:
                indices = [int(i)-1 for i in selection.split(',')]
            
            selected_models = [available_models[i] for i in indices]
            return selected_models
        except (ValueError, IndexError):
            print("Invalid selection. Please try again.")

def parse_args():
    """Parse command-line arguments with comprehensive help information."""
    parser = argparse.ArgumentParser(
        description="""
Seismic Wave Analysis Tool
-------------------------
A tool for analyzing seismic wave propagation through the Earth. Calculate travel 
times, plot ray paths, and generate travel time tables using different velocity models.

Example usage:
  # Calculate P and S wave travel times for a depth of 10 km at 30 degrees distance
  python seismic_analyzer.py --function 1 --depth 10 --distance 30 --phases P,S --model ak135

  # Plot ray paths for multiple phases
  python seismic_analyzer.py --function 2 --depth 100 --distance 45 --phases P,PP,PKP

  # Generate a travel time table
  python seismic_analyzer.py --function 3 --depth 50 --phases P,S
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    function_help = """
Function to run:
  1 = Calculate travel times - Compute arrival times for specified phases
  2 = Plot ray paths - Visualize wave paths through the Earth
  3 = Generate travel time table - Create distance vs. time tables
    """

    phase_help = """
Comma-separated list of seismic phases. Common phases include:

Regional phases (< 15°):
  Pg, Pn - Regional P waves
  Sg, Sn - Regional S waves
  Lg     - Guided crustal waves

Teleseismic phases (> 10°):
  P, S   - Direct P and S waves
  pP, sP - Surface-reflected P waves
  PP, SS - Surface-reflected waves
  PKP, SKS - Core phases

Example: --phases P,S,PP or --phases Pn,Pg,Sg
    """

    model_help = """
Velocity model to use:
  iasp91 - Default global model
  ak135  - Alternative global model (recommended for regional phases)
  prem   - Preliminary Reference Earth Model
    """

    parser.add_argument('--function', type=int, choices=[1, 2, 3],
                       help=function_help)
    parser.add_argument('--depth', type=float,
                       help='Source depth in kilometers (0-6371 km)')
    parser.add_argument('--distance', type=float,
                       help='Distance in degrees (0-180°)')
    parser.add_argument('--phases', type=str,
                       help=phase_help)
    parser.add_argument('--model', type=str, 
                       choices=['iasp91', 'ak135', 'prem'],
                       default='ak135',
                       help=model_help)

    # Add examples group
    example_group = parser.add_argument_group('examples')
    example_group.add_argument(
        '--examples',
        action='store_true',
        help='Show usage examples and exit'
    )

    try:
        args = parser.parse_args()
        
        # If --examples is specified, show examples and exit
        if hasattr(args, 'examples') and args.examples:
            print("""
Usage Examples
-------------
1. Calculate regional phase travel times:
   python seismic_analyzer.py --function 1 --depth 10 --distance 8 --phases Pg,Pn,Sg,Sn --model ak135

2. Plot teleseismic ray paths:
   python seismic_analyzer.py --function 2 --depth 100 --distance 60 --phases P,PP,PKP --model iasp91

3. Generate travel time table for core phases:
   python seismic_analyzer.py --function 3 --depth 200 --phases PKP,PKIKP,SKS --model prem

4. Interactive mode (recommended for new users):
   python seismic_analyzer.py
            """)
            exit(0)
    except Exception as e:
        print(f"Error parsing arguments: {str(e)}")
        parser.print_help()
        exit(1)

    return parser

def main():
    """Main function handling both interactive and command-line modes."""
    parser = parse_args()
    args = parser.parse_args()
    analyzer = SeismicAnalyzer()

    # Command-line mode
    if args.function:
        try:
            if args.depth is None:
                print("Error: --depth is required")
                return
            
            # Validate phases
            if args.phases:
                phases = [p.strip() for p in args.phases.split(',')]
            else:
                print("Error: --phases is required")
                return
            
            # Handle different functions
            if args.function in [1, 2]:  # Travel times or ray paths
                if args.distance is None:
                    print("Error: --distance is required for functions 1 and 2")
                    return
                
                if args.function == 1:
                    analyzer.calculate_travel_times(args.depth, args.distance, phases, args.model)
                else:  # function == 2
                    analyzer.plot_ray_paths(args.depth, args.distance, phases, args.model)
                    
            elif args.function == 3:  # Travel time table
                # Use default distance range if not specified
                start = 0
                end = 180
                step = 10
                distances = np.arange(start, end + step, step)
                analyzer.generate_time_table(args.depth, distances, phases, [args.model])
        except Exception as e:
            print(f"Error: {str(e)}")
        return
            
    # Interactive mode
    while True:
        print("\n=== Seismic Wave Analysis Tool ===")
        print("1. Calculate Travel Times")
        print("2. Plot Ray Paths")
        print("3. Generate Travel Time Table")
        print("4. Exit")
        
        choice = input("\nSelect an option (1-4): ").strip()
        
        if choice == '4':
            print("Goodbye!")
            break
            
        if choice not in ['1', '2', '3']:
            print("Invalid choice. Please select 1-4.")
            continue
            
        try:
            # Get common inputs
            depth = get_float_input("Source depth (km): ", 0, 6371)
            phases = display_phase_menu()
            
            if choice in ['1', '2']:
                distance = get_float_input("Distance (degrees): ", 0, 180)
                print("\nSelect velocity model:")
                model = select_models()[0]  # Use first selected model
                
                if choice == '1':
                    analyzer.calculate_travel_times(depth, distance, phases, model)
                else:  # choice == '2'
                    analyzer.plot_ray_paths(depth, distance, phases, model)
                    
            else:  # choice == '3'
                print("\nDistance range for travel time table:")
                start = get_float_input("Start distance (degrees): ", 0, 180)
                end = get_float_input("End distance (degrees): ", start, 180)
                step = get_float_input("Step size (degrees): ", 1, end-start)
                
                print("\nSelect velocity models to compare:")
                models = select_models()
                
                distances = np.arange(start, end + step, step)
                analyzer.generate_time_table(depth, distances, phases, models)
                
        except Exception as e:
            print(f"\nError: {str(e)}")
        
        input("\nPress Enter to continue...")

if __name__ == "__main__":
    main()