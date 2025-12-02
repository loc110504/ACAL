heading = """
This AI-powered system generates accurate and trustworthy answers for legal questions using argumentation-based reasoning.
The system will:
1. Choost the best team of many legal experts for your task and question
2. Create supporting and challenging arguments for each option
3. Allow you to review and modify arguments (adaptive design for expert/non-expert users)
4. Produce a final answer with explanations
"""

examples = """

### Example Questions for Some Tasks We Have Experimented:

**Task 1: Hearsay**
```
On the issue of whether James is an smart individual, the fact that James came first in his class in law school. Is this a hearsay?
```

**Task 2: Mr. Chen**
```
Mr. Chen, 78 years old, recently widowed, lives in a single-story apartment. He has 
diabetes, moderate hearing loss, and early-stage Parkinson's disease. He is socially 
isolated and has been showing signs of depression. His children live in another city 
and visit monthly.
```

**Task 3: Ms. Rodriguez**
```
Ms. Rodriguez, 85 years old, lives with her daughter who works full-time. She uses 
a walker, has heart disease, and requires assistance with daily activities like 
bathing and meal preparation. She is mentally sharp but physically frail.
```
"""

usage = """
## How to Use This System

### Step 1: Enter Patient Information
Provide comprehensive details about the elderly patient including:
- Age and living situation
- Medical conditions and medications
- Mobility and cognitive status
- Social support and family involvement
- Any recent incidents or concerns

### Step 2: Review Generated Arguments
The system will generate handling options with supporting and challenging arguments.
You can interact with these arguments using simple commands:

**Available Commands:**
- `accept` - Accept all arguments and proceed to final care plan
- `remove [index]` - Remove a specific argument by its index number
- `add support [option_number] [argument_text]` - Add a supporting argument
- `add challenge [option_number] [argument_text]` - Add a challenging argument

**Examples:**
- `remove 3` - Removes argument #3
- `add support 1 The patient has good insurance coverage for home modifications`
- `add challenge 2 The patient's neighborhood lacks accessible public transportation`

### Step 3: Get Your Care Plan
After accepting the arguments, the system will:
- Validate each argument's relevance and accuracy
- Generate a comprehensive care plan
- Provide decision confidence scores
- Explain the reasoning behind recommendations

### Tips for Best Results
- Be specific about medical conditions and functional limitations
- Include information about the patient's preferences and goals
- Consider both physical and psychosocial needs
- Review arguments carefully - they directly influence the final plan
"""