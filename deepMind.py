import os
import json
from datetime import datetime
from mistralai.client import MistralClient
import logging

# Configure logging
logging.basicConfig(
    filename='deepmind_debug.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("DeepMindAgent")

# === File Paths ===
DASHBOARD_RESULTS_PATH = r"C:\Users\dtfygu876\prompt_codes\csvChunking\Chatbot_for_Biosensor"
EMOTIONAL_STATE_FILE = os.path.join(DASHBOARD_RESULTS_PATH, "emotional_state.txt")
IMAGE_ANALYSIS_FILE = os.path.join(DASHBOARD_RESULTS_PATH, "imageAnalysis.txt")
CHART_DATA_FILE = os.path.join(DASHBOARD_RESULTS_PATH, "chart_data.json")

# === Mistral API Keys ===
MISTRAL_KEYS = [
    "jmKl4M4zZsRHdRihUWS1j7jwOcQYsaGV",
    "oixMXymsn6byOMzy9ZfDR5HvZAWljMQr",
    "GeGcfonMU5A7huGpAy1EXw4QJc7T5mTi"
]


def get_mistral_client(task_index: int) -> MistralClient:
    """Get Mistral client with rotated API key"""
    api_key = MISTRAL_KEYS[task_index % len(MISTRAL_KEYS)]
    return MistralClient(api_key=api_key)


# === File Loaders ===
def load_latest_emotional_state():
    """Load the latest emotional state data"""
    if not os.path.exists(EMOTIONAL_STATE_FILE):
        return "Unknown"

    try:
        with open(EMOTIONAL_STATE_FILE, "r") as f:
            lines = f.readlines()
            if not lines:
                return "No data"

            # Find the last complete entry
            latest_block = []
            for line in reversed(lines):
                if line.strip() == "=== Emotional Analysis Entry ===":
                    break
                if line.strip():
                    latest_block.insert(0, line.strip())
            return "\n".join(latest_block) if latest_block else "Unknown"
    except Exception as e:
        logger.error(f"Error reading emotional state: {str(e)}")
        return f"Error: {str(e)}"


def load_latest_image_analysis():
    """Load the latest image analysis data"""
    if not os.path.exists(IMAGE_ANALYSIS_FILE):
        return {"error": "File not found"}

    try:
        with open(IMAGE_ANALYSIS_FILE, "r", encoding="utf-8") as f:
            lines = f.readlines()
            if not lines:
                return {"error": "No entries found"}

            # Find the last valid JSON entry
            for line in reversed(lines):
                if "|" in line:
                    parts = line.split("|")
                    if len(parts) >= 3 and parts[2].startswith("{"):
                        try:
                            return json.loads(parts[2])
                        except json.JSONDecodeError:
                            continue
            return {"error": "No valid JSON entries"}
    except Exception as e:
        logger.error(f"Error reading image analysis: {str(e)}")
        return {"error": str(e)}


# === Prompt Generators ===
def cognitive_load_prompt(emotion: str, image_data: dict) -> str:
    """Generate prompt for cognitive load analysis"""
    return f"""
**Role**: Cognitive Load Analyst
You are an expert in interpreting multimodal data to assess cognitive load.

### Input Data:
**Emotional State**:
{emotion}

**Visual Analysis (VLM)**:
{json.dumps(image_data, indent=2)}

### Analysis Instructions:
1. Interpret emotional state and visual analysis
2. Determine cognitive load level: [Low, Medium, High]
3. Provide brief explanation (1-2 sentences)

### Output Format (JSON):
{{
  "cognitive_load_level": "Low/Medium/High",
  "explanation": "Short paragraph"
}}
"""


def teaching_advice_prompt(emotion: str, task: str, image_data: dict) -> str:
    """Generate prompt for teaching advice"""
    return f"""
You are an AI assistant helping a teacher improve their teaching strategy.
The student is currently doing: **{task}**.

### Input Data:
**Emotional State**:
{emotion}

**Visual Analysis (VLM)**:
{json.dumps(image_data, indent=2)}

### Response Requirements:
- Provide 2-3 concrete suggestions in bullet points
- Focus on improving learning or reducing stress
- Keep response under 100 words
"""


def advisor_prompt(analysis: dict) -> str:
    """Generate prompt for cognitive load advisor"""
    level = analysis.get("cognitive_load_level", "Unknown")
    explanation = analysis.get("explanation", "")

    return f"""
You are an AI teaching Assistant specializing in cognitive load management. 
Your task is to turn cognitive load analysis into short practical actions.

### Cognitive Load Analysis:
Level: {level}
Explanation: {explanation}

### Response Requirements:
- Generate 2-3 concrete, actionable suggestions
- Focus on immediate adjustments the user can make
- Use simple, direct language
- Keep response under 50 words
- Format as bullet points
"""


def summary_prompt(analysis: dict, advisor_response: str, user_message: str) -> str:
    """Generate prompt for student report"""
    return f"""
**Role**: Educational Assessment Specialist
Your task is to generate a comprehensive student report based on multimodal analysis.

### Input Data:
**Cognitive Load Analysis**:
Level: {analysis.get('cognitive_load_level', 'Unknown')}
Explanation: {analysis.get('explanation', 'No explanation')}

**Advisor Recommendations**:
{advisor_response}

**Student Message**:
{user_message}

### Report Requirements:
1. Executive Summary (2-3 sentences)
2. Cognitive Load Assessment
3. Practical Recommendations
4. Teacher Guidance

### Format:
- Use markdown with clear section headings
- Include emojis for visual clarity
- Keep under 150 words
"""


# === Main DeepMindAgent Class ===
class DeepMindAgent:
    def __init__(self):
        self.logger = logger
        self.cognitive_load_history = []
        self.physiological_data = []
        self.chart_data_file = CHART_DATA_FILE

    def get_cognitive_load_analysis(self):
        """Perform cognitive load analysis"""
        try:
            emotion = load_latest_emotional_state()
            image_data = load_latest_image_analysis()

            client = get_mistral_client(0)
            prompt = cognitive_load_prompt(emotion, image_data)

            response = client.chat(
                model="mistral-large-latest",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=400
            )
            content = response.choices[0].message.content

            # Extract JSON from response
            try:
                json_start = content.find('{')
                json_end = content.rfind('}') + 1
                json_str = content[json_start:json_end]
                analysis = json.loads(json_str)

                # Add timestamp and save record
                timestamp = datetime.now().isoformat()
                record = {
                    "timestamp": timestamp,
                    "load_level": analysis.get("cognitive_load_level", "Unknown"),
                    "explanation": analysis.get("explanation", "No explanation")
                }
                self.cognitive_load_history.append(record)
                self._save_chart_data()

                return analysis
            except json.JSONDecodeError:
                self.logger.error("Failed to parse JSON response")
                return {"error": "JSON parsing failed", "raw_response": content}

        except Exception as e:
            self.logger.error(f"Cognitive load analysis error: {str(e)}")
            return {"error": f"Analysis failed: {str(e)}"}

    def get_teacher_advice(self, task_name="Unknown Task"):
        """Get teaching strategies"""
        try:
            emotion = load_latest_emotional_state()
            image_data = load_latest_image_analysis()

            client = get_mistral_client(1)
            prompt = teaching_advice_prompt(emotion, task_name, image_data)

            response = client.chat(
                model="mistral-large-latest",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=200
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            self.logger.error(f"Teacher advice error: {str(e)}")
            return f"⚠️ Error: {str(e)}"

    def get_cognitive_load_advisor(self, analysis: dict):
        """Get practical advice based on cognitive load analysis"""
        try:
            client = get_mistral_client(2)
            prompt = advisor_prompt(analysis)

            response = client.chat(
                model="mistral-large-latest",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=200
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            self.logger.error(f"Advisor error: {str(e)}")
            return f"⚠️ Error: {str(e)}"

    def get_student_report(self, user_message: str):
        """Generate comprehensive student report"""
        try:
            # Get cognitive load analysis
            cl_analysis = self.get_cognitive_load_analysis()

            if "error" in cl_analysis:
                return f"⚠️ Analysis failed: {cl_analysis['error']}"

            # Get advisor recommendations
            advisor_response = self.get_cognitive_load_advisor(cl_analysis)

            # Generate report
            client = get_mistral_client(0)
            prompt = summary_prompt(cl_analysis, advisor_response, user_message)

            response = client.chat(
                model="mistral-large-latest",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=400
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            self.logger.error(f"Report generation error: {str(e)}")
            return f"⚠️ Error: {str(e)}"

    def _save_chart_data(self):
        """Save data for chart generation"""
        try:
            chart_data = {
                "cognitive_load_history": self.cognitive_load_history,
                "last_updated": datetime.now().isoformat()
            }

            with open(self.chart_data_file, "w") as f:
                json.dump(chart_data, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save chart data: {str(e)}")


# === Standalone Test Execution ===
if __name__ == "__main__":
    agent = DeepMindAgent()

    print("\n🔍 [TEST] Cognitive Load Analysis:")
    try:
        analysis = agent.get_cognitive_load_analysis()
        print(json.dumps(analysis, indent=2))
    except Exception as e:
        print(f"❌ Error: {e}")

    print("\n📚 [TEST] Teacher Advice (Sample Task: 'Group Discussion'):")
    try:
        print(agent.get_teacher_advice("Group Discussion"))
    except Exception as e:
        print(f"❌ Error: {e}")

    print("\n🧠 [TEST] Cognitive Load Advisor:")
    try:
        advisor = agent.get_cognitive_load_advisor(analysis)
        print(advisor)
    except Exception as e:
        print(f"❌ Error: {e}")

    print("\n📝 [TEST] Student Report:")
    try:
        print(agent.get_student_report("I'm having trouble with this topic"))
    except Exception as e:
        print(f"❌ Error: {e}")