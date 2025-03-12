import os
from autogen import AssistantAgent
from abc import ABC, abstractmethod
import json
from datetime import datetime
import markdown
import re
from typing import Dict, List, Any
import uuid
from sar_project.agents.base_agent import SARBaseAgent
import google.generativeai as genai
from tavily import TavilyClient

from sar_project.knowledge.knowledge_base import KnowledgeBase


class OperationsSectionChiefAgent(SARBaseAgent):
    def __init__(self):
        system_message = """
        You are the Operations Section Chief for Search and Rescue operations.
        Your responsibilities include:
        - Analyzing incident information
        - Making tactical decisions
        - Assigning and coordinating search, rescue, and medical teams
        - Ensuring safety protocols are followed
        - Adapting to changing conditions
        - Managing resources effectively
        """
        super().__init__(
            name="Operations_Chief",
            role="Operations Section Chief",
            system_message=system_message,
            knowledge_base=KnowledgeBase()
        )
        self.required_fields = [
            "location",
            "missing_persons",
            "environmental_conditions",
            "available_resources"
        ]

    # The main function called to allow processing of the request
    def process_request(self, message: str, incident_id: str, location: str, is_initial= True) -> List[Dict]:
        """
        Main method to process incoming requests and generate team assignments
        """
    
        try:
            
            # Preprocess and Search using Tavily
            tavily_data = self.searchRelevantInfo(message, incident_id, location)
            if "weather" in tavily_data.keys():
                message += f"weather_forecast {json.dumps(tavily_data["weather"])}"
            if "terrain_info" in tavily_data.keys(): 
                message += f"terrain_info {json.dumps(tavily_data["terrain_info"])}"

            # Process the operation
            prompt = self._generate_llm_prompt(incident_id, message, is_initial)
            llm_response = self._send_to_llm(prompt)
            self._update_knowledge_base(incident_id, llm_response)
            self._create_docs(llm_response, output_filename=incident_id)
            return llm_response
            
        except Exception as e:
            raise Exception(f"Error processing SAR operation: {str(e)}")

    # The function to preprocess the message using Tavily
    def searchRelevantInfo(self, message: str, incident_id: str, location) -> Dict:
        # Try using TAVILY API to get weather forecast and recommended search route
        client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
        # extract the location from the message string
        
        
        query1 = f"What is the weather forecast for {location} tomorrow?"
        query2 = f"Find all trails/roads near {location} prioritizing open paths, avoiding steep terrain, and ensuring multiple search points are covered."
        # Fetch weather details to predict conditions during the search
        weather_response = client.search(
            query=query1,
            time_range="day",
            include_answer="basic"
        )
        current_weather = {"description": weather_response["answer"], "results": json.dumps(weather_response["results"])}

        # Fetch trail information to plan the search route
        # Grabs images that we can use as well
        route_response = client.search(
            query=query2,
            time_range="day",
            include_images=True,
            include_image_descriptions=True,
            include_answer="basic"
        )
        current_route_info = {'description': route_response["answer"], "images": route_response["images"], "results": json.dumps(route_response["results"]) }
        
        self.kb.update_weather(location, current_weather, incident_id)
        self.kb.update_terrain(location, current_route_info, incident_id)

        return {"weather": current_weather, "terrain_info": current_route_info}

    # updating the knowledge base
    def _update_knowledge_base(self, incident_id: str, data: Dict) -> None:
        """
        Update the knowledge base with the incident data and team assignments
        """
        # Update the mission history with the incident data
        self.kb.log_mission_event({
            "incident_id": incident_id,
            "timestamp": datetime.now().isoformat(),
            "data": data
        })

    # Generate the prompt for the LLM based on whether this is an initial incident or not   
    def _generate_llm_prompt(self, incident_id: str, incident_data: Dict, is_initial: bool = True) -> str:
        """Generate a structured prompt for the LLM that requests JSON-formatted output"""
        if not is_initial:
            prev_info = self.kb.mission_history
            prev_info = list(filter(lambda item: "incident_id" in item and item["incident_id"] == incident_id, self.kb.mission_history))
            # limit to just the last incident, to reduce size on updates?
            prev_info = prev_info[-1] if len(prev_info) > 0 else {}

            return f"""
            You are an AI assistant for the Search and Rescue Operations Section Chief. Your role is to analyze incident information and generate tactical assignments for search, rescue, and medical teams.
            Given the previous incident information:
            {prev_info} 
            and the following update:
            {incident_data}

            Analyze the update and perform the requested action. Use the previous information as needed, prioritizing the latest information. Provide your response in the following JSON format:

            Example:

            Previous Mission History:
            ```json
            
            {{
                "entry_id": {{incident_id}},
                "time": "timestamp",
                "location": "N 34.05, W 118.24",
                "description": "Search initiated for missing hiker.",
                "status": "Active",
                "teams_involved": ["Team Alpha"],
                "resources_used": ["Dog Unit 1"],
                "notes": ""
            }}
            
            ```

            If there is a new team created make sure to add: 
            
            ```json 
            "team_assignments": [
            {{
                "team_id": "TEAM-DESIGNATION",
                "team_type": "SEARCH|RESCUE|MEDICAL",
                "leader": "Team leader name",
                "team_size" : <number>,
                "objective": "Primary objective statement",
                "tasks": ["task1", "task2", "..."],
                "equipment": ["equipment1", "equipment2", "..."],
                "reporting": "reporting_frequency",
                "priority": "HIGH|MEDIUM|LOW"
            }},
            // Additional teams as needed
            ```

            Ensure that your response includes the appropriate teams involved and important information from the update. Leave missing information as <>.

            """
        
        return f"""
    You are an AI assistant for the Search and Rescue Operations Section Chief. Your role is to analyze incident information and generate tactical assignments for search, rescue, and medical teams.

    Given the following incident information:
    {json.dumps(incident_data, indent=2)}

    Please analyze the situation and provide your response in the following JSON format:

    ```json
    {{
        "report": {{
                    "situation": {{
                        "location": "",
                        "missing_persons": {{"name": "", "additioanl_info": ""}}, 
                        "environmental_conditions": {{
                            "temperature": None,
                            "wind_speed": None,
                            "visibility": None,
                            "precipitation": None,
                            "hazards": []
                        }}
                    }},
                    "available_resources": {{
                        "search_teams": <number>,
                        "rescue_teams": <number>,
                        "medical_teams": <number>,
                        "equipment": ["eqipment1", "equipment2", "..."]
                    }},
                    "additional_information": "",
                    "priority_updates": []
                    "route_images" : ["image1", "image2", "..."]
        }}, 
        "analysis": {{
            "severity_level": "HIGH|MEDIUM|LOW",
            "situation_summary": "Brief description of the overall situation",
            "primary_risks": ["risk1", "risk2", "..."],
            "resource_requirements": {{
                "search_teams_needed": <number>,
                "rescue_teams_needed": <number>,
                "medical_teams_needed": <number>
            }}
        }},
        "strategic_decisions": {{
            "search_strategy": "Description of search approach",
            "resource_allocation": "How resources should be distributed",
            "priority_areas": ["area1", "area2", "..."],
            "timeline": "Expected operation timeline"
        }},
        "team_assignments": [
            {{
                "team_id": "TEAM-DESIGNATION",
                "team_type": "SEARCH|RESCUE|MEDICAL",
                "leader": "Team leader name",
                "team_size" : <number>,
                "objective": "Primary objective statement",
                "tasks": ["task1", "task2", "..."],
                "equipment": ["equipment1", "equipment2", "..."],
                "reporting": "reporting_frequency",
                "priority": "HIGH|MEDIUM|LOW"
            }},
            // Additional teams as needed
        ],
        "contingency_plans": {{
            "weather_deterioration": "Actions if weather worsens",
            "medical_emergency": "Medical evacuation procedures",
            "communications_failure": "Backup communication methods",
            "resource_limitations": "How to handle resource shortages"
        }}
    }}
    ```

    Ensure your response includes:
    1. A detailed situation analysis with severity level and key risks
    2. Clear strategic decisions for resource allocation and search approach
    3. Specific team assignments with all required details
    4. Comprehensive contingency plans for various scenarios

    Your analysis should prioritize both operational effectiveness and team safety. Do not make up information and leave missing information as <>. 
    """

    # Actually calling the LLM to generate the response
    def generate(self, prompt: str) -> str:
        """Generate response from the LLM"""

        model = genai.GenerativeModel("gemini-2.0-flash")

        response = model.generate_content(
            prompt
        )

        return response

    # Send the prompt to the LLM and process the response
    def _send_to_llm(self, prompt: str) -> Dict:
        """Send prompt to LLM and process response"""
        try:
            # Use the inherited AssistantAgent's LLM configuration
            response = self.generate(prompt)
            
            # Parse the response into the expected format
            # Note: This might need adjustment based on actual LLM response format
            parsed_response = self._parse_llm_response(response.text)
            
            return parsed_response
            
        except Exception as e:
            raise Exception(f"Error in LLM processing: {str(e)}")

    # Parse the llm response (if json or text (should be json though))
    def _parse_llm_response(self, response: str) -> Dict:
        """Parse the LLM response into structured format, handling both JSON and text formats"""
        try:
            # Extract JSON if response contains it - look for JSON between ```json and ``` markers
            json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
            
            if json_match:
                # Parse the JSON content
                try:
                    parsed_data = json.loads(json_match.group(1))
                    # Validate expected structure
                    # self._validate_response_structure(parsed_data)
                    return parsed_data
                except json.JSONDecodeError:
                    print("Warning: Found JSON markers but couldn't parse content. Falling back to text parsing.")
            
            # If no valid JSON found, attempt to parse structured text
            return self._parse_text_response(response)
        
        except Exception as e:
            print(f"Error parsing LLM response: {str(e)}")
            # Return a minimal valid structure
            return self._create_fallback_response()

    # Validate the response structure to ensure it has the expected format
    def _validate_response_structure(self, data: Dict) -> None:
        """Validate that the parsed data has the expected structure"""
        required_sections = ['analysis', 'strategic_decisions', 'team_assignments', 'contingency_plans']
        
        for section in required_sections:
            if section not in data:
                raise ValueError(f"Missing required section: {section}")
        
        # Validate team assignments
        if not isinstance(data['team_assignments'], list) or len(data['team_assignments']) == 0:
            raise ValueError("Team assignments must be a non-empty list")
        
        # Check each team assignment has required fields
        team_required_fields = ['team_id', 'team_type', 'objective', 'tasks']
        if "team_assignment" in data :
            for team in data['team_assignments']:
                for field in team_required_fields:
                    if field not in team:
                        raise ValueError(f"Team assignment missing required field: {field}")

    # Parse the text response into structured format, this is if the response doesn't have a json format
    def _parse_text_response(self, response: str) -> Dict:
        """Extract structured information from a text response"""
        # Initialize with default structure
        parsed_data = {
            "analysis": {
                "severity_level": "MEDIUM",
                "situation_summary": "",
                "primary_risks": [],
                "resource_requirements": {
                    "search_teams_needed": 0,
                    "rescue_teams_needed": 0,
                    "medical_teams_needed": 0
                }
            },
            "strategic_decisions": {
                "search_strategy": "",
                "resource_allocation": "",
                "priority_areas": [],
                "timeline": ""
            },
            "team_assignments": [],
            "contingency_plans": {
                "weather_deterioration": "",
                "medical_emergency": "",
                "communications_failure": "",
                "resource_limitations": ""
            }
        }
        
        # Extract sections
        analysis_section = self._extract_section(response, "SITUATION ANALYSIS", "STRATEGIC DECISIONS")
        strategic_section = self._extract_section(response, "STRATEGIC DECISIONS", "TEAM ASSIGNMENTS")
        teams_section = self._extract_section(response, "TEAM ASSIGNMENTS", "CONTINGENCY PLANS")
        contingency_section = self._extract_section(response, "CONTINGENCY PLANS", None)
        
        # Parse analysis section
        if analysis_section:
            # Extract severity level
            severity_match = re.search(r'severity[:\s]+([A-Za-z]+)', analysis_section, re.IGNORECASE)
            if severity_match:
                parsed_data["analysis"]["severity_level"] = severity_match.group(1).upper()
            
            # Extract primary risks
            risk_matches = re.findall(r'(?:risk|hazard|danger)[s]?[:\s]+(.*?)(?:\n|$)', analysis_section, re.IGNORECASE)
            if risk_matches:
                risks = []
                for match in risk_matches:
                    # Split by common separators and clean up
                    items = re.split(r'[,;]', match)
                    risks.extend([item.strip() for item in items if item.strip()])
                parsed_data["analysis"]["primary_risks"] = risks
            
            # Extract resource requirements
            search_match = re.search(r'(\d+)\s+search\s+team', analysis_section, re.IGNORECASE)
            if search_match:
                parsed_data["analysis"]["resource_requirements"]["search_teams_needed"] = int(search_match.group(1))
            
            rescue_match = re.search(r'(\d+)\s+rescue\s+team', analysis_section, re.IGNORECASE)
            if rescue_match:
                parsed_data["analysis"]["resource_requirements"]["rescue_teams_needed"] = int(rescue_match.group(1))
            
            medical_match = re.search(r'(\d+)\s+medical\s+team', analysis_section, re.IGNORECASE)
            if medical_match:
                parsed_data["analysis"]["resource_requirements"]["medical_teams_needed"] = int(medical_match.group(1))
        
        # Parse team assignments
        if teams_section:
            # Look for team blocks like "Team Alpha" or "Search Team 1"
            team_blocks = re.split(r'\n\s*(?:Team|TEAM)\s+[A-Za-z0-9]+', teams_section)
            if len(team_blocks) > 1:
                # First element is typically intro text, so skip it
                for i, block in enumerate(team_blocks[1:], 1):
                    team_name_match = re.search(r'(?:Team|TEAM)\s+([A-Za-z0-9]+)', teams_section.split(block)[0])
                    if team_name_match:
                        team_id = team_name_match.group(1)
                    else:
                        team_id = f"TEAM-{i}"
                    
                    team_type_match = re.search(r'(SEARCH|RESCUE|MEDICAL)', block, re.IGNORECASE)
                    team_type = team_type_match.group(1).upper() if team_type_match else "UNKNOWN"
                    
                    leader_match = re.search(r'[Ll]eader[:\s]+(.*?)(?:\n|$)', block)
                    leader = leader_match.group(1).strip() if leader_match else ""
                    
                    objective_match = re.search(r'[Oo]bjective[:\s]+(.*?)(?:\n|$)', block)
                    objective = objective_match.group(1).strip() if objective_match else ""
                    
                    # Extract tasks (usually bullet points or numbered items)
                    tasks = re.findall(r'(?:[-*•]|\d+\.)\s+(.*?)(?:\n|$)', block)
                    if not tasks:
                        # Try alternative patterns
                        tasks = re.findall(r'(?<=\n)\s*((?:[A-Z][^.\n]*\.)|(?:[A-Za-z]+ +[a-z][^.\n]*\.))', block)
                    
                    # Extract equipment
                    equipment_match = re.search(r'[Ee]quipment[:\s]+(.*?)(?:\n|$)', block)
                    equipment = []
                    if equipment_match:
                        equipment_text = equipment_match.group(1)
                        # Split by commas or other separators
                        equipment = [item.strip() for item in re.split(r'[,;]', equipment_text) if item.strip()]
                    
                    # Extract reporting frequency
                    reporting_match = re.search(r'[Rr]eport(?:ing)?[:\s]+(.*?)(?:\n|$)', block)
                    reporting = reporting_match.group(1).strip() if reporting_match else "standard"
                    
                    # Determine priority
                    priority = "HIGH" if re.search(r'urgent|critical|high priority', block, re.IGNORECASE) else "NORMAL"
                    
                    # Add team to assignments
                    parsed_data["team_assignments"].append({
                        "team_id": team_id,
                        "team_type": team_type,
                        "leader": leader,
                        "objective": objective,
                        "tasks": tasks,
                        "equipment": equipment,
                        "reporting": reporting,
                        "priority": priority
                    })
        
        return parsed_data

    # Extract a section of text between start_marker and end_marker(used to extract json sections)
    def _extract_section(self, text: str, start_marker: str, end_marker: str = None) -> str:
        """Extract a section of text between start_marker and end_marker"""
        # Find the start position
        start_pos = text.find(start_marker)
        if start_pos == -1:
            return ""
        
        # Move past the start marker
        start_pos += len(start_marker)
        
        # Find the end position
        if end_marker:
            end_pos = text.find(end_marker, start_pos)
            if end_pos == -1:
                return text[start_pos:].strip()
            return text[start_pos:end_pos].strip()
        else:
            return text[start_pos:].strip()

    # Default response structure if parsing fails (might change this to just error)
    def _create_fallback_response(self) -> Dict:
        """Create a minimal valid response structure as fallback"""
        return {
            "analysis": {
                "severity_level": "MEDIUM",
                "situation_summary": "Information could not be parsed properly",
                "primary_risks": ["unknown"],
                "resource_requirements": {
                    "search_teams_needed": 1,
                    "rescue_teams_needed": 1,
                    "medical_teams_needed": 1
                }
            },
            "strategic_decisions": {
                "search_strategy": "Default strategy - require manual review",
                "resource_allocation": "Deploy minimum resources until situation clarified",
                "priority_areas": ["last known location"],
                "timeline": "Immediate response required"
            },
            "team_assignments": [
                {
                    "team_id": "SEARCH-DEFAULT",
                    "team_type": "SEARCH",
                    "leader": "Team Leader",
                    "objective": "Locate missing persons",
                    "tasks": ["Begin standard search pattern", "Report findings immediately"],
                    "equipment": ["standard_gear"],
                    "reporting": "every_30_min",
                    "priority": "HIGH"
                }
            ],
            "contingency_plans": {
                "weather_deterioration": "Seek shelter and await instructions",
                "medical_emergency": "Contact base for emergency extraction",
                "communications_failure": "Return to last known checkpoint",
                "resource_limitations": "Prioritize life-saving operations"
            }
        }

    # Create the markdown docs
    def _create_docs(self, data: Dict, output_filename: str = "output") -> None:
        """Convert JSON data to a Markdown file"""
        # Create a document for the overall incident without the team assignments
        # filter out the team_assignments data
        overall_data = {key: value for key, value in data.items() if key != "team_assignments"}
        self._create_markdown(overall_data, output_filename)

        # Seperate into documents for each of the team leaders
        if "team_assignments" in data:
            for team in data["team_assignments"]:
                team_id = team["team_id"]
                team_data = {
                    "team_assignments": [team]
                }
                self._create_markdown(team_data, output_filename, f"{team_id}_{output_filename}")

    # Create the markdown file from json
    def _create_markdown(self,json_data, search_dir, output_filename="output"):
        """
        Converts JSON data to a Markdown file.

        Args:
            json_data: A Python dictionary or a JSON string.
            output_filename: The name of the Markdown file to create.
        """
        output_dir = "./src/sar_project/output_docs/"
        os.makedirs(output_dir, exist_ok=True) # Create directory if it doesn't exist
        os.makedirs(f"{output_dir}{search_dir}/", exist_ok=True) # Create directory if it doesn't exist

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_filename = f"{output_dir}{search_dir}/{output_filename}_{timestamp}.md"  # Append timestamp to filename
        if isinstance(json_data, str):
            try:
                data = json.loads(json_data)  # Parse JSON string if it's a string
            except json.JSONDecodeError:
                print("Invalid JSON string.")
                return

        elif isinstance(json_data, dict):
            data = json_data  # Already a dictionary

        else:
            print("Invalid input: Must be a JSON string or a Python dictionary.")
            return

        markdown_content = ""

        def _generate_markdown(data, level=0): # Recursive helper function
            nonlocal markdown_content # Access the outer scope's markdown_content

            if isinstance(data, dict):
                for key, value in data.items():
                    markdown_content += f"{'#' * (level + 1)} {key}\n"  # Heading
                    _generate_markdown(value, level + 1) # Recurse for nested dictionaries

            elif isinstance(data, list):
                for item in data:
                    markdown_content += "- " # List item
                    _generate_markdown(item, level + 1) # Recurse for nested data

            elif isinstance(data, (str, int, float, bool, type(None))): # Base cases
                markdown_content += str(data) + "\n" # Add value as text

            else: # Handle other data types if needed
                markdown_content += str(data) + "\n"  # Default string conversion

        _generate_markdown(data) # Start the recursive function

        try:
            # Convert Markdown to HTML (optional but often useful)
            html = markdown.markdown(markdown_content)
            with open(output_filename, "w", encoding="utf-8") as f:
                f.write(markdown_content) # Write the markdown content

            print(f"Markdown file '{output_filename}' created successfully.")

        except Exception as e:
            print(f"An error occurred: {e}")

    # Get the knowledge base access to public
    def get_knowledge_base(self):
        return self.kb