import json
import os
from dotenv import load_dotenv
import google.generativeai as genai
from sar_project.agents.operation_chief_agent import OperationsSectionChiefAgent

load_dotenv()  # Load environment variables from .env file

# Configure the Google Generative AI API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


def run_example(operations_chief, incident_id, initial_input, is_initial=True):
    try:
        operations_chief.process_request(initial_input, incident_id, is_initial)
    except Exception as e:
        print(f"Error: {str(e)}")

def example_01():
    initial_input = """
    Date: February 18, 2025
    Time of Departure: 8:00 AM
    Location of Trail: Green Hollow Trail, Blue Ridge Mountains
    Hikers:
    Alice Turner – 28 years old, experienced hiker
    Ben Jackson – 32 years old, moderate experience
    Carlos Mendoza – 24 years old, novice hiker
    Trail Information: The Green Hollow Trail is a 12-mile loop with a 3,000-foot elevation gain, known for its dense forest areas and steep ascents. The trailhead is located near the base of the Blue Ridge Mountains, approximately 20 miles from the nearest town, Millbrook.
    Planned Return Time: 2:00 PM
    Last Contact Time: 2:10 PM – The hikers were seen at the 6-mile mark by another group hiking in the opposite direction. They mentioned they would take a break near the summit and then head down.
    Weather Forecast:
    Morning: Clear skies, mild temperatures around 60°F
    Afternoon: A slight chance of rain by 4:00 PM, becoming overcast by 5:00 PM, with temperatures dropping to around 50°F.
    Emergency Contact:
    The hikers' families were notified at 5:30 PM when they did not return. A call was made to local authorities at 6:00 PM.    
    
    Team Breakdown: 3 search teams: Team Alpha, Team Bravo, Team Charlie, 1 medical team: Team Delta, 1 rescue team: Team Echo. 
    Available Equipment: Radios, GPS Devices, First Aid Kits, Emergency Supplies
    Contact Info for leaders:
    Team Alpha: Sarah Adams, 555-123-4567
    Team Bravo: Mark Roberts, 555-234-5678
    Team Charlie: Lisa Chang, 555-345-6789
    Team Delta: Chris Thompson, 555-456-7890
    Team Echo: Rachel Lee, 555-567-8901
    """

    update_1 = """
    Update: 8:30 PM – Local authorities have initiated a search operation but have not located the missing hikers. The weather has deteriorated with heavy rain and strong winds. The temperature has dropped to 45°F.
    """

    update_2 = """
    Update: 10:00 PM – The search operation has been temporarily suspended due to severe weather conditions. The local authorities are regrouping and planning to resume the search at first light. The temperature is now 40°F with heavy rain and strong winds.
    """

    update_3 = """
    Update: 6:00 AM – The search operation has resumed with improved weather conditions. The temperature is 38°F with light rain and fog. The teams are focusing on the upper section of the trail near the summit.
    """

    return "test_id_01", [initial_input, update_1, update_2, update_3]

def example_02():
    initial_input = """
    Location: Mount Baker Wilderness
    Situation: 3 hiking groups (8 people) reported missing after sudden storm
    Conditions: 28°F, winds 25mph, visibility 0.5 miles, Light Rain
    Current status: Cell towers showing last signals from 3 different sectors
    Teams: 2 search teams, 1 rescue team, 1 medical team
    Available Equipment: Radios, FPS Devices, First Aid Kits, 
    """
    return "test_id_02", initial_input

def example_03():
    initial_input = """
    Location: Mount Hood National Forest
    Situation: 5 climbers reported missing after avalanche
    Conditions: 32°F, winds 15mph, visibility 0.5 miles, Snowing
    Current status: Cell towers showing last signals from 2 different sectors
    Teams: 3 search teams, 2 rescue teams, 1 medical team
    Available Equipment: Radios, GPS Devices, First Aid Kits, Emergency Supplies
    """

    return "test_id_03", initial_input

# Example usage
if __name__ == "__main__":
    # Initialize the Operations Section Chief agent
    operations_chief = OperationsSectionChiefAgent()
    
    id, input = example_02()
    run_example(operations_chief, id, input)

    id, input = example_03()
    run_example(operations_chief, id, input)

    id, inputs = example_01()
    run_example(operations_chief, id, inputs[0])
    for i in range(1, len(inputs)):
        run_example(operations_chief, id, inputs[i], is_initial=False)
    
    print(operations_chief.get_knowledge_base().mission_history)


