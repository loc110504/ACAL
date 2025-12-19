import unittest
from state import Argument, GraphState
from node import (
    rag_retrieval,
    overall_options,
    agent_selector,
    multi_agent_argument_generator,
    human_review,
    route_after_human_review,
    argument_validator,
    final_answer_generator
)

class TestFullLegalWorkflow(unittest.TestCase):
    def setUp(self):
        self.state = {
            'task_name': 'hearsay',
            'task_info': 'Alex is being prosecuted for participation in a criminal conspiracy. To prove that Alex participated in the conspiracy, the prosecution\'s witness testifies that she heard Alex making plans to meet with his co-conspirators.',
            'enable_streaming': False
        }

    def test_full_workflow(self):
        state = self.state.copy()
        # Step 1: Retrieval
        state = rag_retrieval(state)
        # Step 2: Options
        state = overall_options(state)
        # Step 3: Agent selection
        state = agent_selector(state, 'support')
        state = agent_selector(state, 'attack')
        # Step 4: Argument generation
        state = multi_agent_argument_generator(state)
        # Step 5: Human review (simulate complete)
        state = human_review(state)
        route = route_after_human_review(state)
        # Step 6: Argument validation
        state = argument_validator(state)
        # Step 7: Final answer
        state = final_answer_generator(state)
        print('\nFinal Answer:', state['final_answer']['answer'])

if __name__ == '__main__':
    unittest.main()
