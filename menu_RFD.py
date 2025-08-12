import shlex
import json
import os
import subprocess

# --- Configuration ---
# Universal config file for all tools
CONFIG_FILE = 'config.json'
# RFdiffusion-specific file for saved run commands
RUNS_HISTORY_FILE = 'saved_runs_rfd.json'

# This dictionary defines the arguments for each potential type.
# Added 'help_text' for more intuitive user guidance.
POTENTIAL_DEFINITIONS = {
    'monomer_ROG': {
        'help_text': 'Radius of Gyration for a single chain. Encourages compactness.',
        'weight': {'default': 1, 'type': 'float', 'prompt': 'Weight for the potential'},
        'min_dist': {'default': 15, 'type': 'float', 'prompt': 'Minimum distance for ROG calculation'}
    },
    'binder_ROG': {
        'help_text': 'Radius of Gyration for a binder chain. Encourages the binder to be compact.',
        'binderlen': {'default': None, 'type': 'int', 'required': True, 'prompt': 'Length of the binder chain'},
        'weight': {'default': 1, 'type': 'float', 'prompt': 'Weight for the potential'},
        'min_dist': {'default': 15, 'type': 'float', 'prompt': 'Minimum distance for ROG calculation'}
    },
    'dimer_ROG': {
        'help_text': 'Radius of Gyration for a dimeric complex. Encourages overall compactness of two chains.',
        'binderlen': {'default': None, 'type': 'int', 'required': True, 'prompt': 'Length of one monomer chain (binderlen)'},
        'weight': {'default': 1, 'type': 'float', 'prompt': 'Weight for the potential'},
        'min_dist': {'default': 15, 'type': 'float', 'prompt': 'Minimum distance for ROG calculation'}
    },
    'binder_ncontacts': {
        'help_text': 'Number of contacts within a binder. Encourages a well-folded binder.',
        'binderlen': {'default': None, 'type': 'int', 'required': True, 'prompt': 'Length of the binder chain'},
        'weight': {'default': 1, 'type': 'float', 'prompt': 'Weight for the potential'},
        'r_0': {'default': 8, 'type': 'float', 'prompt': 'r_0 for contact calculation'},
        'd_0': {'default': 4, 'type': 'float', 'prompt': 'd_0 for contact calculation'}
    },
    'interface_ncontacts': {
        'help_text': 'Number of contacts between chains. Encourages a strong binding interface.',
        'binderlen': {'default': None, 'type': 'int', 'required': True, 'prompt': 'Length of the binder chain'},
        'weight': {'default': 1, 'type': 'float', 'prompt': 'Weight for the potential'},
        'r_0': {'default': 8, 'type': 'float', 'prompt': 'r_0 for contact calculation'},
        'd_0': {'default': 6, 'type': 'float', 'prompt': 'd_0 for contact calculation'}
    },
    'monomer_contacts': {
        'help_text': 'Number of contacts within a single chain monomer.',
        'weight': {'default': 1, 'type': 'float', 'prompt': 'Weight for the potential'},
        'r_0': {'default': 8, 'type': 'float', 'prompt': 'r_0 for contact calculation'},
        'd_0': {'default': 2, 'type': 'float', 'prompt': 'd_0 for contact calculation'}
    },
    'olig_contacts': {
        'help_text': 'Contact potential for oligomers, with separate weights for intra- and inter-chain contacts.',
        'contact_matrix': {'default': None, 'type': 'str', 'required': True, 'prompt': 'Contact matrix (advanced, provide manually)'},
        'weight_intra': {'default': 1, 'type': 'float', 'prompt': 'Weight for intra-chain contacts'},
        'weight_inter': {'default': 1, 'type': 'float', 'prompt': 'Weight for inter-chain contacts'},
        'r_0': {'default': 8, 'type': 'float', 'prompt': 'r_0 for contact calculation'},
        'd_0': {'default': 2, 'type': 'float', 'prompt': 'd_0 for contact calculation'}
    },
    'substrate_contacts': {
        'help_text': 'Potential to guide interactions with a substrate.',
        'weight': {'default': 1, 'type': 'float', 'prompt': 'Weight for the potential'},
        's': {'default': 1, 'type': 'float', 'prompt': 'Attractive strength (s)'},
        'r_0': {'default': 8, 'type': 'float', 'prompt': 'Attractive radius (r_0)'},
        'rep_s': {'default': 2, 'type': 'float', 'prompt': 'Repulsive strength (rep_s)'},
        'rep_r_0': {'default': 5.0, 'type': 'float', 'prompt': 'Repulsive radius (rep_r_0)'},
    },
}

# --- Utility Functions ---
def initialize_config():
    """Checks for the universal config file and creates a default if it doesn't exist."""
    if not os.path.exists(CONFIG_FILE):
        print("First time setup: Creating default universal config.json file.")
        print("Please review and edit your preferences using menu option 5.")
        default_config = {
            "user_name": "default_user",
            "RFD": {
                "run_script_path": "scripts/run_inference.py",
                "default_input_dir": "inputs/",
                "default_output_dir": "outputs/"
            }
            # Future tools can be added here, e.g., "AlphaFold": {...}
        }
        with open(CONFIG_FILE, 'w') as f:
            json.dump(default_config, f, indent=4)
    
    # Handle RFD-specific saved runs file
    if not os.path.exists(RUNS_HISTORY_FILE):
        print(f"Creating empty {RUNS_HISTORY_FILE} for RFdiffusion runs.")
        with open(RUNS_HISTORY_FILE, 'w') as f:
            json.dump({}, f)

    return load_config()

def load_config():
    """Loads the configuration from the JSON file."""
    try:
        with open(CONFIG_FILE, 'r') as f:
            config = json.load(f)
            # Ensure the RFD section exists for backward compatibility
            if 'RFD' not in config:
                print("Updating config to new format. Please verify settings.")
                config['RFD'] = {
                    "run_script_path": config.get("run_script_path", "scripts/run_inference.py"),
                    "default_input_dir": config.get("default_input_dir", "inputs/"),
                    "default_output_dir": config.get("default_output_dir", "outputs/")
                }
                # Clean up old top-level keys
                for key in ["run_script_path", "default_input_dir", "default_output_dir"]:
                    if key in config:
                        del config[key]
                save_config(config)
            return config
    except (FileNotFoundError, json.JSONDecodeError):
        print("Error loading config file. Re-initializing.")
        if os.path.exists(CONFIG_FILE):
            os.remove(CONFIG_FILE)
        return initialize_config()


def save_config(config_data):
    """Saves data to the config file."""
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config_data, f, indent=4)
    print("Configuration saved successfully.")

def load_saved_runs():
    """Loads the run history from the RFD-specific JSON file."""
    if not os.path.exists(RUNS_HISTORY_FILE):
        return {}
    try:
        with open(RUNS_HISTORY_FILE, 'r') as f:
            content = f.read()
            if not content:
                return {}
            return json.loads(content)
    except json.JSONDecodeError:
        print(f"Warning: Could not decode {RUNS_HISTORY_FILE}. It might be corrupted. Starting fresh.")
        return {}


def save_run(name, command):
    """Saves a new run command to the RFD-specific history file."""
    runs = load_saved_runs()
    runs[name] = command
    with open(RUNS_HISTORY_FILE, 'w') as f:
        json.dump(runs, f, indent=4)
    print(f"RFD Run '{name}' saved successfully.")

# --- UI and Input Functions ---
def print_menu():
    """Displays the main menu to the user."""
    print("\nRFdiffusion Run Manager")
    print("------------------------")
    print("1. Load and run a saved RFD command")
    print("2. Enter an RFD command from scratch")
    print("3. Build RFD command using a guided template")
    print("4. Delete all saved RFD runs")
    print("5. Edit Preferences")
    print("0. Exit")

def get_user_input(param_details, param_name=""):
    """
    Prompts the user for a single parameter value, printing help text and handling validation.
    """
    is_required = param_details.get('required', False)
    default_val = param_details['default']
    param_type = param_details['type']
    prompt_text = param_details['prompt']
    help_text = param_details.get('help_text')

    if help_text:
        print(f"\n[INFO] {help_text}")

    while True:
        prompt_msg = f"- {prompt_text}\n  ({'Required' if is_required else 'Optional'}, Default: {default_val}). Enter new value or press Enter: "
        user_input = input(prompt_msg).strip()

        if not user_input:
            value = default_val
            if is_required and value is None:
                print("This field is required. Please enter a value.")
                continue
        else:
            try:
                if param_type == 'int': value = int(user_input)
                elif param_type == 'float': value = float(user_input)
                elif param_type == 'bool': value = user_input.lower() in ['true', 't', 'yes', 'y', '1']
                else: value = user_input
            except ValueError:
                print(f"Invalid input. Please enter a value of type '{param_type}'.")
                continue
        
        if 'contig' in param_name or 'inpaint' in param_name or 'hotspot' in param_name:
            if isinstance(value, str) and value and not value.startswith('['):
                value = f'[{value}]'

        return value

def review_and_run_command(command_str):
    """
    Allows the user to review, edit, and then choose to save, run, or do both.
    """
    print("\n" + "="*50)
    print("[DONE] Command generation complete! Review and confirm.")
    print("="*50 + "\n")
    
    final_command = command_str
    try:
        import readline
        readline.set_startup_hook(lambda: readline.insert_text(command_str))
        final_command = input("Final command (edit if needed, then press Enter):\n")
        readline.set_startup_hook()
    except (ImportError, AttributeError):
        print("Your system doesn't support inline editing. Please review carefully.")
        print(f"Generated command: {command_str}")
        edited_command = input("If correct, press Enter. Otherwise, paste edited command:\n")
        final_command = edited_command or command_str

    while True:
        action = input("\nAction: [R]un, [S]ave only, [B]oth (Save & Run), [C]ancel to menu: ").strip().lower()

        should_save = action in ['s', 'b', 'save only', 'both']
        should_run = action in ['r', 'b', 'run', 'both']
        
        if should_save:
            run_name = input("Enter a name for this RFD run: ").strip()
            if run_name:
                save_run(run_name, final_command)
            else:
                print("Save cancelled: No name provided.")
            
            if action == 's':
                print("Command saved. Returning to main menu.")
                return 

        if should_run:
            print("\n--- EXECUTING COMMAND ---")
            try:
                subprocess.run(final_command, shell=True, check=True)
                print("--- EXECUTION COMPLETE ---")
            except subprocess.CalledProcessError as e:
                print(f"--- ERROR DURING EXECUTION ---")
                print(f"Command failed with exit code {e.returncode}")
            except FileNotFoundError:
                print(f"--- ERROR: Command not found ---")
                print(f"Please ensure the script path in your config.json is correct and executable.")
            except Exception as e:
                print(f"An unexpected error occurred: {e}")
            
            print("\nReturning to main menu...")
            return

        if action in ['c', 'cancel']:
            print("Action cancelled. Returning to main menu.")
            return

        if not should_save and not should_run:
             print("Invalid option. Please choose R, S, B, or C.")


# --- Submenu Builders ---
# Note: These builders are currently RFD-specific but could be generalized in the future.
def build_inpainting_submenu():
    """A guided submenu for setting inpainting parameters."""
    print("\n" + "-"*40)
    print("Optional Step: Inpainting")
    print("Specify regions of a protein to 'inpaint' or re-design, for either sequence or structure.")
    if input("Add inpainting details? (y/n): ").strip().lower() != 'y':
        return {}
    
    settings = {}
    inpaint_params = {
        'inpaint_seq': {'default': None, 'type': 'str', 'prompt': 'Regions to inpaint sequence for (e.g., "A1-10,A15-25")', 'help_text': 'Provide a range of residues where the amino acid sequence should be redesigned.'},
        'inpaint_str': {'default': None, 'type': 'str', 'prompt': 'Regions to inpaint structure for (e.g., "B165-178")', 'help_text': 'Provide a range of residues where the 3D structure should be redesigned.'},
    }
    for name, details in inpaint_params.items():
        value = get_user_input(details, f'contigmap.{name}')
        if value:
            settings[f'contigmap.{name}'] = value
    return settings

def build_potentials_submenu():
    """A guided submenu for setting potential parameters."""
    print("\n" + "-"*40)
    print("Optional Step: Guiding Potentials")
    print("Potentials 'steer' the model to favor certain geometric properties (e.g., compactness, contacts).")
    if input("Add guiding potentials? (y/n): ").strip().lower() != 'y':
        return {}
        
    all_potential_strings = []
    
    while True:
        print("\nAvailable potential types:")
        for i, pot_name in enumerate(POTENTIAL_DEFINITIONS.keys(), 1):
            print(f"  {i}. {pot_name}")
        
        choice = input("Select a potential to add (or press Enter to finish): ").strip()
        if not choice:
            break

        try:
            potential_name = list(POTENTIAL_DEFINITIONS.keys())[int(choice) - 1]
            potential_config = POTENTIAL_DEFINITIONS[potential_name]
            print(f"\n--- Configuring '{potential_name}' ---")
            if potential_config.get('help_text'):
                print(f"[INFO] {potential_config['help_text']}")
        except (ValueError, IndexError):
            print("Invalid selection.")
            continue

        potential_args = [f"type:{potential_name}"]
        potential_details = POTENTIAL_DEFINITIONS[potential_name]
        for arg_name, arg_details in potential_details.items():
            if arg_name == 'help_text': continue
            value = get_user_input(arg_details)
            if value is not None and (value != arg_details['default'] or arg_details.get('required')):
                potential_args.append(f"{arg_name}:{value}")
        
        all_potential_strings.append(",".join(potential_args))
        print(f"\n[+] Added potential: {all_potential_strings[-1]}")

    if not all_potential_strings:
        return {}
    
    settings = {}
    settings['potentials.guiding_potentials'] = json.dumps(all_potential_strings)

    print("\n--- Overall Potential Settings ---")
    overall_potential_params = {
        'guide_scale': {'default': 10, 'type': 'float', 'prompt': 'Overall scale for guiding potentials', 'help_text': 'A multiplier for the strength of all active potentials.'},
        'guide_decay': {'default': 'constant', 'type': 'str', 'prompt': 'Decay type for guide strength', 'help_text': 'How the potential strength fades over diffusion time (e.g., constant, linear, quadratic, cubic).'}
    }
    
    for param, details in overall_potential_params.items():
        value = get_user_input(details)
        if value != details['default']:
            settings[f'potentials.{param}'] = value

    return settings

def build_ppi_submenu():
    """A guided submenu for setting PPI hotspot parameters."""
    print("\n" + "-"*40)
    print("Optional Step: Protein-Protein Interface (PPI) Hotspots")
    print("Specify key residues that must be part of the binding interface.")
    if input("Add PPI hotspot constraints? (y/n): ").strip().lower() != 'y':
        return {}
    
    settings = {}
    hotspot_details = {'default': None, 'type': 'str', 'prompt': 'Hotspot residues (e.g., "B28,B29")', 'help_text': 'A comma-separated list of residues (ChainID + Residue Number) that are critical for the interaction.'}
    value = get_user_input(hotspot_details, 'ppi.hotspot_res')
    if value:
        settings['ppi.hotspot_res'] = value
    return settings

def build_advanced_submenu():
    """A guided submenu for setting advanced, less-common parameters."""
    PARAMETERS_ADVANCED = {
        'inference': {
            'ckpt_override_path': {'default': None, 'type': 'str', 'prompt': 'Override model checkpoint path', 'help_text': 'Use a specific, custom-trained model checkpoint file instead of the default.'},
            'symmetry': {'default': None, 'type': 'str', 'prompt': 'Symmetry type (e.g., c2, c3)', 'help_text': 'Apply cyclic symmetry to the design.'},
            'recenter': {'default': True, 'type': 'bool', 'prompt': 'Recenter the motif?', 'help_text': 'Automatically re-center the input motif in the coordinate system.'},
            'radius': {'default': 10.0, 'type': 'float', 'prompt': 'Radius for model neighbors', 'help_text': 'The radius used to select neighboring residues for the model.'},
            'model_only_neighbors': {'default': False, 'type': 'bool', 'prompt': 'Model only neighbors?', 'help_text': 'If true, the model will only see residues within the radius, ignoring other context.'},
            'write_trajectory': {'default': True, 'type': 'bool', 'prompt': 'Write trajectory files?', 'help_text': 'Save the full diffusion trajectory for each design, useful for debugging.'},
            'cautious': {'default': True, 'type': 'bool', 'prompt': 'Run in cautious mode?', 'help_text': 'Use more memory-intensive but safer operations to avoid errors.'},
            'deterministic': {'default': True, 'type': 'bool', 'prompt': 'Use deterministic sampling?', 'help_text': 'Ensures that running the same command twice produces the exact same output.'},
        },
        'diffuser': {
            'T': {'default': 50, 'type': 'int', 'prompt': 'Number of diffusion timesteps', 'help_text': 'The number of "noise" steps in the diffusion process. More steps can lead to better results but take longer.'},
            'partial_T': {'default': None, 'type': 'int', 'prompt': 'Partial diffusion time (e.g., 20)', 'help_text': 'Start the diffusion process from a specified timestep, not from pure noise.'},
        }
    }
    print("\n" + "-"*40)
    print("Optional Step: Advanced Settings")
    if input("Configure advanced settings? (y/n): ").strip().lower() != 'y':
        return {}
    
    settings = {}
    for section, params in PARAMETERS_ADVANCED.items():
        print(f"\n--- {section.upper()} Advanced Settings ---")
        for param, details in params.items():
            full_param_name = f"{section}.{param}"
            value = get_user_input(details, full_param_name)
            if value is not None and value != details['default']:
                settings[full_param_name] = value
    return settings


def edit_preferences():
    """Allows the user to view and edit the RFD section of the config.json file."""
    print("\n--- Edit RFdiffusion Preferences ---")
    config = load_config()
    rfd_config = config.get("RFD", {})
    
    print("Current RFD settings:")
    for key, value in rfd_config.items():
        print(f"  {key}: {value}")
        
    if input("\nDo you want to edit these settings? (y/n): ").strip().lower() != 'y':
        return

    new_rfd_config = {}
    for key, value in rfd_config.items():
        new_value = input(f"Enter new value for '{key}' (or press Enter to keep '{value}'): ").strip()
        new_rfd_config[key] = new_value if new_value else value
    
    config["RFD"] = new_rfd_config
    save_config(config)

# --- Main Application Logic ---
def build_command_from_template():
    """Walks the user through building a command using the interactive parameter guide."""
    config = load_config()
    rfd_config = config.get("RFD", {}) # Use the RFD-specific config
    
    parameters = {
        'inference': {
            'input_pdb': {'default': None, 'type': 'str', 'required': False, 'prompt': f'Input PDB path', 'help_text': f'Path to a starting PDB file, if any. This is relative to your input directory: {rfd_config.get("default_input_dir")}'},
            'num_designs': {'default': 10, 'type': 'int', 'prompt': 'Number of designs to generate', 'help_text': 'The total number of unique protein structures the model will attempt to create.'},
            'output_prefix': {'default': 'output/design', 'type': 'str', 'required': True, 'prompt': f'Output prefix', 'help_text': f'A name for your output files. This will be placed in your output directory: {rfd_config.get("default_output_dir")}'},
        },
        'contigmap': {
            'contigs': {'default': None, 'type': 'str', 'required': True, 'prompt': 'Contig string (e.g., 100-200 or A10-25/5-10)', 'help_text': 'This is the core of your design. It specifies chain breaks, scaffolded regions, and lengths of motifs to build.'},
        },
    }

    print("\n" + "="*50)
    print(">> RFdiffusion Guided Command Builder <<")
    print("="*50)
    print("\nI will walk you through creating a command step-by-step.")
    print("Press Enter at any prompt to accept the default value shown.")
    
    user_settings = {}
    step_counter = 1

    for section, params in parameters.items():
        print(f"\n--- Step {step_counter}: {section.upper()} Settings ---")
        step_counter += 1
        for param, details in params.items():
            full_param_name = f"{section}.{param}"
            value = get_user_input(details, full_param_name)
            
            if value:
                if full_param_name == 'inference.input_pdb' and not os.path.isabs(value):
                    value = os.path.join(rfd_config.get("default_input_dir", ""), value)
                elif full_param_name == 'inference.output_prefix' and not os.path.isabs(os.path.dirname(value)):
                     value = os.path.join(rfd_config.get("default_output_dir", ""), value)
            
            if value is not None:
                user_settings[full_param_name] = value
    
    # Optional Submenus
    user_settings.update(build_inpainting_submenu())
    user_settings.update(build_potentials_submenu())
    user_settings.update(build_ppi_submenu())
    user_settings.update(build_advanced_submenu())

    base_script = rfd_config.get("run_script_path", "")
    command_parts = [base_script]
    for key, value in user_settings.items():
        if value is None: continue
        
        formatted_value = shlex.quote(str(value))
        command_parts.append(f"{key}={formatted_value}")
    
    review_and_run_command(" ".join(command_parts))

def load_and_run():
    """Loads a saved RFD command and allows the user to review and run it."""
    runs = load_saved_runs()
    if not runs:
        print("No saved RFD runs found.")
        return
    
    print("\n--- Saved RFD Runs ---")
    run_keys = list(runs.keys())
    for i, name in enumerate(run_keys, 1):
        print(f"{i}. {name}")
    
    try:
        choice = input(f"Select a run to load (1-{len(run_keys)}): ").strip()
        key = run_keys[int(choice) - 1]
        command = runs[key]
        print(f"\nLoaded command for '{key}':")
        review_and_run_command(command)
    except (ValueError, IndexError):
        print("Invalid selection.")

def delete_all_runs():
    """Deletes the saved RFD runs file after confirmation."""
    if os.path.exists(RUNS_HISTORY_FILE):
        confirm = input(f"Are you sure you want to delete ALL saved RFD runs from {RUNS_HISTORY_FILE}? (y/n): ").strip().lower()
        if confirm == 'y':
            try:
                os.remove(RUNS_HISTORY_FILE)
                print("Deleted all saved RFD run data.")
                with open(RUNS_HISTORY_FILE, 'w') as f:
                    json.dump({}, f)
            except OSError as e:
                print(f"Error deleting file: {e}")
    else:
        print("No saved RFD runs file to delete.")

def main():
    """Main function to run the command-line interface."""
    try:
        config = initialize_config()
        print(f"Welcome, {config.get('user_name', 'user')}!")
        
        while True:
            # This menu is now specific to the RFD tool.
            # A higher-level menu could be added in main.py to select the tool first.
            print_menu()
            choice = input("Select an option (0-5): ").strip()

            if choice == '1':
                load_and_run()
            elif choice == '2':
                command = input("Paste or type your full RFdiffusion command here:\n")
                if command:
                    review_and_run_command(command)
            elif choice == '3':
                build_command_from_template()
            elif choice == '4':
                delete_all_runs()
            elif choice == '5':
                edit_preferences()
            elif choice == '0':
                print("Exiting RFdiffusion Run Manager. Goodbye!")
                break
            else:
                print("Invalid input. Please enter a number between 0 and 5.")
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user. Exiting.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        print("Exiting.")
