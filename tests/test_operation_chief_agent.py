import unittest
import os
import json
from datetime import datetime
from unittest.mock import patch, MagicMock
from sar_project.agents.operation_chief_agent import OperationsSectionChiefAgent  # Replace with your actual path
from sar_project.knowledge.knowledge_base import KnowledgeBase # Replace with your actual path

class TestOperationsSectionChiefAgent(unittest.TestCase):

    def setUp(self):
        self.agent = OperationsSectionChiefAgent()
        self.test_incident_id = "test_incident_123"
        self.location = "Mt. Diablo State Park"
        self.test_incident_data = """
            "location": "Mt. Diablo State Park",
            "missing_persons": [{"name": "John Doe", "additional_info": "Last seen hiking"}],
            "environmental_conditions": {"temperature": 70, "wind_speed": 10, "visibility": "Good", "precipitation": "None", "hazards": []},
            "available_resources": {"search_teams": 3, "rescue_teams": 2, "medical_teams": 1, "equipment": ["ropes", "radios"]}
        """
        self.test_update_data = {"""Situation Overview:
            Search and rescue operations are underway for John Doe, who was last seen hiking in the area. Current environmental conditions remain favorable, with a temperature of 70°F, wind speeds at 10 mph, good visibility, and no precipitation. No significant hazards have been reported in the area.

            Operational Update:

            Search Teams: Three search teams have been deployed and are actively covering key hiking trails and surrounding areas.
            Rescue Teams: Two rescue teams are on standby to assist in case of discovery or emergency extraction.
            Medical Support: One medical team is prepared to provide immediate care if needed.
            Equipment Usage: Teams are utilizing ropes and radios for coordination and navigation in the terrain.
        """}
        # We'll use the previous data for now
        self.test_prev_data = {      
            "location": "Mt. Diablo State Park",
            "missing_persons": [{"name": "John Doe", "additional_info": "Last seen hiking"}],
            "environmental_conditions": {"temperature": 70, "wind_speed": 10, "visibility": "Good", "precipitation": "None", "hazards": []},
            "available_resources": {"search_teams": 3, "rescue_teams": 2, "medical_teams": 1, "equipment": ["ropes", "radios"]}
        }
        self.mock_kb = KnowledgeBase()
        self.mock_kb.log_mission_event({
            "incident_id": self.test_incident_id,
            "timestamp": datetime.now().isoformat(),
            "data": self.test_prev_data
        })
        self.agent.kb = self.mock_kb # Assign the mock to the agent

    def tearDown(self):
        # Clean up generated files after tests if needed
        output_dir = "./src/sar_project/output_docs/"
        if os.path.exists(output_dir):
            for root, dirs, files in os.walk(output_dir):
                for file in files:
                    if file.startswith(self.test_incident_id):
                        os.remove(os.path.join(root, file))
                    if file.endswith(".md"):
                        os.remove(os.path.join(root, file))
                for dir in dirs:
                    try:
                        os.rmdir(os.path.join(root, dir))
                    except OSError as e:
                        print(f"Directory not empty. Skipping: {e}")

    @patch("sar_project.agents.operation_chief_agent.OperationsSectionChiefAgent._send_to_llm")
    def test_process_request_error(self, mock_send_to_llm):
        mock_send_to_llm.side_effect = Exception("LLM Error")
        with self.assertRaisesRegex(Exception, "Error processing SAR operation: LLM Error"):
            self.agent.process_request(self.test_incident_data, self.test_incident_id, self.location)

    @patch("sar_project.agents.operation_chief_agent.OperationsSectionChiefAgent.generate")
    def test_send_to_llm_success(self, mock_generate):
        mock_generate.return_value = MagicMock(text='```json\n{"test": "response"}\n```')
        response = self.agent._send_to_llm("test prompt")
        self.assertEqual(response, {"test": "response"})

    @patch("sar_project.agents.operation_chief_agent.OperationsSectionChiefAgent.generate")
    def test_send_to_llm_error(self, mock_generate):
        mock_generate.side_effect = Exception("LLM Error")
        with self.assertRaisesRegex(Exception, "Error in LLM processing: LLM Error"):
            self.agent._send_to_llm("test prompt")


    def test_parse_llm_response_json(self):
        response = '```json\n{"analysis": {"severity_level": "HIGH"}, "strategic_decisions": {}, "team_assignments": [{"team_id": "T1", "team_type": "S", "objective": "O1", "tasks": ["t1"]}], "contingency_plans": {}}\n```'
        parsed_response = self.agent._parse_llm_response(response)
        self.assertEqual(parsed_response["analysis"]["severity_level"], "HIGH")
        self.assertEqual(parsed_response["team_assignments"][0]["team_id"], "T1")

    def test_parse_llm_response_text(self):
        response = "SITUATION ANALYSIS:\nseverity: HIGH\nrisk: Fire\n1 search team\nTEAM ASSIGNMENTS:\nTeam Alpha\nleader: John\nobjective: Search\n- task 1\nCONTINGENCY PLANS:"
        parsed_response = self.agent._parse_llm_response(response)
        self.assertEqual(parsed_response["analysis"]["severity_level"], "HIGH")
        self.assertEqual(parsed_response["analysis"]["primary_risks"], ["Fire"])
        self.assertEqual(parsed_response["analysis"]["resource_requirements"]["search_teams_needed"], 1)
        self.assertEqual(parsed_response["team_assignments"][0]["team_id"], "Alpha")
        self.assertEqual(parsed_response["team_assignments"][0]["objective"], "Search")
        self.assertEqual(parsed_response["team_assignments"][0]["tasks"], ["task 1"])

    def test_parse_llm_response_no_json_or_text(self):
        response = "Some random text"
        parsed_response = self.agent._parse_llm_response(response)
        self.assertEqual(parsed_response["analysis"]["severity_level"], "MEDIUM")  # Check fallback

    def test_validate_response_structure_valid(self):
        data = {"analysis": {}, "strategic_decisions": {}, "team_assignments": [{"team_id": "T1", "team_type": "S", "objective": "O1", "tasks": ["t1"]}], "contingency_plans": {}}
        self.agent._validate_response_structure(data)  # Should not raise an exception

    def test_validate_response_structure_missing_section(self):
        data = {"analysis": {}, "strategic_decisions": {}, "team_assignments": [{"team_id": "T1", "team_type": "S", "objective": "O1", "tasks": ["t1"]}]}  # Missing contingency_plans
        with self.assertRaises(ValueError) as context:
            self.agent._validate_response_structure(data)
        self.assertIn("Missing required section: contingency_plans", str(context.exception))

    def test_validate_response_structure_empty_team_assignments(self):
        data = {"analysis": {}, "strategic_decisions": {}, "team_assignments": [], "contingency_plans": {}}  # Empty team_assignments
        with self.assertRaises(ValueError) as context:
            self.agent._validate_response_structure(data)
        self.assertIn("Team assignments must be a non-empty list", str(context.exception))


