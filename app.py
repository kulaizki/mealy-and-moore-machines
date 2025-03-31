import streamlit as st
import graphviz
import numpy as np

def main():
    st.title("Finite-State Machine Visualizer")
    st.write("A tool to practice and visualize Mealy and Moore machines for automata theory")

    # Nav 
    app_mode = st.sidebar.selectbox(
        "Choose Mode",
        ["Introduction", "Moore Machine", "Mealy Machine", "Practice Mode"]
    )

    if app_mode == "Introduction":
        show_introduction()
    elif app_mode == "Moore Machine":
        show_moore_machine()
    elif app_mode == "Mealy Machine":
        show_mealy_machine()
    elif app_mode == "Practice Mode":
        show_practice_mode()

def show_introduction():
    st.header("Introduction to Finite-State Machines")

    st.subheader("Moore Machines")
    st.write("""
    A Moore machine is a finite-state machine where the output depends only on the current state.

    **Definition**: A Moore machine is a 6-tuple (Q, Σ, Δ, δ, λ, q₀) where:
    - Q is a finite set of states
    - Σ is a finite set of input symbols
    - Δ is a finite set of output symbols
    - δ: Q × Σ → Q is the transition function
    - λ: Q → Δ is the output function
    - q₀ ∈ Q is the initial state

    In a Moore machine, the output is associated with the state itself.
    """)

    st.subheader("Mealy Machines")
    st.write("""
    A Mealy machine is a finite-state machine where the output depends on both the current state and the input.

    **Definition**: A Mealy machine is a 6-tuple (Q, Σ, Δ, δ, λ, q₀) where:
    - Q is a finite set of states
    - Σ is a finite set of input symbols
    - Δ is a finite set of output symbols
    - δ: Q × Σ → Q is the transition function
    - λ: Q × Σ → Δ is the output function
    - q₀ ∈ Q is the initial state

    In a Mealy machine, the output is associated with the transition between states.
    """)

    st.subheader("Differences and Applications")
    st.write("""
    - **Moore machines** produce output based solely on the current state, making them simpler in some cases. The output length is typically len(input) + 1.
    - **Mealy machines** can often represent the same behavior with fewer states. The output length is typically len(input).
    - Both are used in digital circuit design, protocol specification, and computational linguistics.
    """)

    st.subheader("How to use this app")
    st.write("""
    1. Navigate to the Moore or Mealy machine tab to create and visualize your own state machines.
    2. Use the Practice Mode to test your understanding of state machine behavior.
    3. Input transitions using the provided forms and see the visual representation immediately.
    4. Test your machines with input strings to see the step-by-step execution and output sequence.
    """)

def show_moore_machine():
    st.header("Moore Machine Visualizer")
    st.write("Define and visualize a Moore machine. Remember, output depends only on the state.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Machine Definition")

        # Number of states
        num_states = st.number_input("Number of States", min_value=1, max_value=10, value=3, key="moore_num_states")

        # Input alphabet
        input_alphabet_str = st.text_input("Input Alphabet (comma-separated)", "0,1", key="moore_input_alpha")
        inputs = [x.strip() for x in input_alphabet_str.split(",") if x.strip()]
        if not inputs:
            st.warning("Please define at least one input symbol.")
            return

        # Output alphabet
        output_alphabet_str = st.text_input("Output Alphabet (comma-separated)", "A,B", key="moore_output_alpha")
        outputs = [x.strip() for x in output_alphabet_str.split(",") if x.strip()]
        if not outputs:
            st.warning("Please define at least one output symbol.")
            return

        state_names = [f"q{i}" for i in range(num_states)]

        initial_state = st.selectbox("Initial State", state_names, key="moore_initial_state")

        # State outputs
        st.subheader("State Outputs (λ: Q → Δ)")
        state_outputs = {}
        cols_per_row = 5
        num_rows = (num_states + cols_per_row - 1) // cols_per_row
        for r in range(num_rows):
            cols = st.columns(min(cols_per_row, num_states - r * cols_per_row))
            for i in range(len(cols)):
                state_index = r * cols_per_row + i
                state_name = state_names[state_index]
                with cols[i]:
                    state_outputs[state_name] = st.selectbox(f"Output for {state_name}", outputs, key=f"moore_out_{state_name}")

        # Transition table
        st.subheader("Transition Table (δ: Q × Σ → Q)")
        transitions = {}
        # Use columns for inputs for a more standard table layout
        header_cols = st.columns(len(inputs) + 1)
        header_cols[0].write("**State**")
        for idx, inp in enumerate(inputs):
             header_cols[idx + 1].write(f"**Input: {inp}**")

        for i in range(num_states):
            current_state = state_names[i]
            transitions[current_state] = {}
            row_cols = st.columns(len(inputs) + 1)
            row_cols[0].write(current_state)
            for j, inp in enumerate(inputs):
                key = f"moore_trans_{current_state}_{inp}"
                with row_cols[j + 1]:
                    dest_state = st.selectbox(
                        f"δ({current_state}, {inp})",
                        state_names,
                        label_visibility="collapsed", # Hide label as it's redundant with header
                        key=key
                    )
                    transitions[current_state][inp] = dest_state

    with col2:
        st.subheader("Visualization")

        # Create a graphviz visualization
        g = graphviz.Digraph()
        g.attr(rankdir="LR")

        # Add states with outputs
        for state_name in state_names:
            output = state_outputs.get(state_name, "?") # Use get for safety
            label = f"{state_name}\nOutput: {output}"

            # Mark initial state
            if state_name == initial_state:
                g.node(state_name, label=label, shape="doublecircle")
            else:
                g.node(state_name, label=label, shape="circle")

        # Add transitions
        for from_state, trans_dict in transitions.items():
            # Aggregate edges for the same destination
            edges = {}
            for input_sym, to_state in trans_dict.items():
                if to_state not in edges:
                    edges[to_state] = []
                edges[to_state].append(input_sym)

            for to_state, labels in edges.items():
                g.edge(from_state, to_state, label=", ".join(labels))

        # Render the graph
        st.graphviz_chart(g)

        # Allow the user to test the machine
        st.subheader("Test Machine")
        test_input = st.text_input("Input String (e.g., " + "".join(inputs*2) + ")", "".join(inputs), key="moore_test_input")

        if st.button("Run Machine", key="moore_run"):
            if not initial_state:
                st.error("Please define an initial state.")
            elif not state_outputs:
                 st.error("Please define state outputs.")
            elif not transitions:
                 st.error("Please define transitions.")
            elif test_input is not None: # Allow empty string test
                current_state = initial_state
                output_sequence = []
                state_sequence = [current_state]

                # Initial state output
                if current_state in state_outputs:
                    output_sequence.append(state_outputs[current_state])
                else:
                    st.error(f"Output for initial state {current_state} not defined!")
                    return # Stop processing

                st.write(f"**Initial State:** {current_state}")
                st.write(f"**Initial Output:** {state_outputs.get(current_state, 'ERROR')}")
                st.write("---")

                step_by_step = []
                step_by_step.append(f"Start at state **{current_state}**, output **{state_outputs.get(current_state, '?')}**")

                valid_run = True
                for i, symbol in enumerate(test_input):
                    if symbol not in inputs:
                        st.error(f"Input symbol '{symbol}' at position {i} is not in the defined input alphabet {inputs}!")
                        valid_run = False
                        break

                    if current_state not in transitions or symbol not in transitions[current_state]:
                        st.error(f"No transition defined for state **{current_state}** with input **{symbol}**!")
                        valid_run = False
                        break

                    next_state = transitions[current_state][symbol]

                    if next_state not in state_outputs:
                        st.error(f"Output for destination state **{next_state}** (from state **{current_state}** on input **{symbol}**) is not defined!")
                        valid_run = False
                        break

                    next_output = state_outputs[next_state]
                    step_by_step.append(f"Input **'{symbol}'**: **{current_state}** → **{next_state}**, output **{next_output}**")
                    current_state = next_state
                    output_sequence.append(next_output)
                    state_sequence.append(current_state)

                st.write("**Execution Trace:**")
                for step in step_by_step:
                    st.write(step)
                st.write("---")

                if valid_run:
                    st.write(f"**Final State:** {current_state}")
                    st.write(f"**State Sequence:** {' → '.join(state_sequence)}")
                    st.write(f"**Output Sequence:** {''.join(output_sequence)}")


def show_mealy_machine():
    st.header("Mealy Machine Visualizer")
    st.write("Define and visualize a Mealy machine. Remember, output depends on state AND input.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Machine Definition")

        # Number of states
        num_states = st.number_input("Number of States", min_value=1, max_value=10, value=3, key="mealy_num_states")

        # Input alphabet
        input_alphabet_str = st.text_input("Input Alphabet (comma-separated)", "0,1", key="mealy_input_alpha")
        inputs = [x.strip() for x in input_alphabet_str.split(",") if x.strip()]
        if not inputs:
            st.warning("Please define at least one input symbol.")
            return

        # Output alphabet
        output_alphabet_str = st.text_input("Output Alphabet (comma-separated)", "A,B", key="mealy_output_alpha")
        outputs = [x.strip() for x in output_alphabet_str.split(",") if x.strip()]
        if not outputs:
            st.warning("Please define at least one output symbol.")
            return

        # State names
        state_names = [f"q{i}" for i in range(num_states)]

        # Initial state
        initial_state = st.selectbox("Initial State", state_names, key="mealy_initial_state")

        # Transition and output table
        st.subheader("Transition & Output Table (δ: Q × Σ → Q, λ: Q × Σ → Δ)")
        transitions = {}
        transition_outputs = {}

        # Header row
        header_cols = st.columns(len(inputs) * 2 + 1) # State + (Next State, Output) for each input
        header_cols[0].write("**State**")
        for i, inp in enumerate(inputs):
             header_cols[i*2 + 1].write(f"**δ(*, {inp})**")
             header_cols[i*2 + 2].write(f"**λ(*, {inp})**")
             st.markdown("""<hr style="height:1px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True) # visual separator

        # Input rows
        for i in range(num_states):
            current_state = state_names[i]
            transitions[current_state] = {}
            transition_outputs[current_state] = {}

            row_cols = st.columns(len(inputs) * 2 + 1)
            row_cols[0].write(current_state)

            for j, inp in enumerate(inputs):
                 with row_cols[j*2 + 1]:
                    dest_state = st.selectbox(
                        f"Next state for q{i}, input {inp}",
                        state_names,
                        label_visibility="collapsed",
                        key=f"mealy_trans_{current_state}_{inp}"
                    )
                 with row_cols[j*2 + 2]:
                    trans_output = st.selectbox(
                        f"Output for q{i}, input {inp}",
                        outputs,
                        label_visibility="collapsed",
                        key=f"mealy_out_{current_state}_{inp}"
                    )
                 transitions[current_state][inp] = dest_state
                 transition_outputs[current_state][inp] = trans_output

    with col2:
        st.subheader("Visualization")

        # Create a graphviz visualization
        g = graphviz.Digraph()
        g.attr(rankdir="LR")

        # Add states
        for state_name in state_names:
            # Mark initial state
            if state_name == initial_state:
                g.node(state_name, shape="doublecircle")
            else:
                g.node(state_name, shape="circle")

        # Add transitions with outputs
        for from_state, trans_dict in transitions.items():
            for input_sym, to_state in trans_dict.items():
                # Ensure output is defined before accessing
                output = "?"
                if from_state in transition_outputs and input_sym in transition_outputs[from_state]:
                    output = transition_outputs[from_state][input_sym]
                else:
                    st.warning(f"Output for transition ({from_state}, {input_sym}) not defined yet.")

                g.edge(from_state, to_state, label=f"{input_sym} / {output}")

        # Render the graph
        st.graphviz_chart(g)

        # Allow the user to test the machine
        st.subheader("Test Machine")
        test_input = st.text_input("Input String (e.g., " + "".join(inputs*2) + ")", "".join(inputs), key="mealy_test_input")

        if st.button("Run Machine", key="mealy_run"):
            if not initial_state:
                st.error("Please define an initial state.")
            elif not transitions:
                 st.error("Please define transitions.")
            elif not transition_outputs:
                 st.error("Please define transition outputs.")
            elif test_input is not None: # Allow empty string test
                current_state = initial_state
                output_sequence = []
                state_sequence = [current_state]

                st.write(f"**Initial State:** {current_state}")
                st.write("---")

                step_by_step = []
                step_by_step.append(f"Start at state **{current_state}**")

                valid_run = True
                for i, symbol in enumerate(test_input):
                    if symbol not in inputs:
                        st.error(f"Input symbol '{symbol}' at position {i} is not in the defined input alphabet {inputs}!")
                        valid_run = False
                        break

                    if current_state not in transitions or symbol not in transitions[current_state]:
                        st.error(f"No transition defined for state **{current_state}** with input **{symbol}**!")
                        valid_run = False
                        break

                    if current_state not in transition_outputs or symbol not in transition_outputs[current_state]:
                        st.error(f"No output defined for state **{current_state}** with input **{symbol}**!")
                        valid_run = False
                        break

                    next_state = transitions[current_state][symbol]
                    output = transition_outputs[current_state][symbol]

                    step_by_step.append(f"Input **'{symbol}'**: **{current_state}** → **{next_state}**, output **{output}**")
                    current_state = next_state
                    output_sequence.append(output)
                    state_sequence.append(current_state)

                st.write("**Execution Trace:**")
                for step in step_by_step:
                    st.write(step)
                st.write("---")

                if valid_run:
                    st.write(f"**Final State:** {current_state}")
                    st.write(f"**State Sequence:** {' → '.join(state_sequence)}")
                    st.write(f"**Output Sequence:** {''.join(output_sequence)}")


def show_practice_mode():
    st.header("Practice Mode")
    st.write("Test your understanding by analyzing and working with predefined machines")

    # --- Session State Initialization ---
    # Ensure essential keys exist for practice mode
    if "practice_seed" not in st.session_state:
        st.session_state["practice_seed"] = np.random.randint(1000)
    if "practice_type" not in st.session_state:
        st.session_state["practice_type"] = "Identify Machine Type" # Default
    if "practice_difficulty" not in st.session_state:
        st.session_state["practice_difficulty"] = "Easy" # Default
    #------------------------------------

    col1, col2 = st.columns([3, 1]) 

    with col1:
        practice_type = st.selectbox(
            "Select Practice Type",
            ["Identify Machine Type", "Predict Output", "Complete Transitions"],
            index=["Identify Machine Type", "Predict Output", "Complete Transitions"].index(st.session_state.practice_type), # Preserve selection across runs
            key="practice_type_select"
        )
        difficulty = st.radio(
            "Difficulty Level",
            ["Easy", "Medium", "Hard"],
            index=["Easy", "Medium", "Hard"].index(st.session_state.practice_difficulty), # Preserve selection
            key="practice_difficulty_select",
            horizontal=True
        )

    with col2:
        st.write("") # Spacer
        st.write("") # Spacer
        if st.button("🔄 New Problem"):
            # Update session state with current selections before generating new seed
            st.session_state.practice_type = st.session_state.practice_type_select
            st.session_state.practice_difficulty = st.session_state.practice_difficulty_select
            # Generate new seed and clear previous answers
            st.session_state["practice_seed"] = np.random.randint(1000)
            # Clear specific keys related to answers for all practice types
            keys_to_clear = ["answer_submitted", "user_answer", "correct_answer", "user_output",
                             "complete_check_result", "identify_answer", "predict_answer"]
            # Also clear keys related to user inputs for the 'complete' problem
            for key in list(st.session_state.keys()):
                if key.startswith("complete_to_q") or key.startswith("complete_output_q"):
                    keys_to_clear.append(key)

            for key in keys_to_clear:
                 if key in st.session_state:
                    del st.session_state[key]
            st.rerun() # Use st.rerun() for better state handling


    # Use the stored or newly selected type/difficulty
    current_practice_type = st.session_state.practice_type
    current_difficulty = st.session_state.practice_difficulty
    seed = st.session_state.practice_seed

    st.write(f"**Problem Type:** {current_practice_type} | **Difficulty:** {current_difficulty} | **Seed:** {seed}")
    st.divider()

    np.random.seed(seed) # Ensure problem is reproducible until 'New Problem' is clicked

    # Create practice problem based on type and difficulty
    if current_practice_type == "Identify Machine Type":
        create_identify_machine_problem(current_difficulty)
    elif current_practice_type == "Predict Output":
        create_predict_output_problem(current_difficulty)
    elif current_practice_type == "Complete Transitions":
        create_complete_transitions_problem(current_difficulty)

def create_identify_machine_problem(difficulty):
    st.subheader("Identify Machine Type")
    st.write("Determine whether the given machine is a Moore or Mealy machine based on its diagram.")

    # Generate a random machine
    is_mealy = np.random.choice([True, False])

    if difficulty == "Easy":
        num_states = 2
        input_alphabet = ["0", "1"]
    elif difficulty == "Medium":
        num_states = 3
        input_alphabet = ["0", "1"]
    else:  # Hard
        num_states = 4
        input_alphabet = ["0", "1", "a"] # Mix numbers and letters

    output_alphabet = ["A", "B"]
    if difficulty == "Hard":
        output_alphabet.append("C")

    state_names = [f"q{i}" for i in range(num_states)]
    initial_state = state_names[0] # Always start at q0 for simplicity

    # Create the machine visualization
    g = graphviz.Digraph()
    g.attr(rankdir="LR")

    # Add states
    state_outputs = {} # Only used for Moore

    for i in range(num_states):
        state_name = state_names[i]

        if is_mealy:
            # For Mealy, no output on the states
            shape = "doublecircle" if state_name == initial_state else "circle"
            g.node(state_name, shape=shape)
        else:
            # For Moore, add output to states
            output = np.random.choice(output_alphabet)
            state_outputs[state_name] = output
            label = f"{state_name}\n{output}"
            shape = "doublecircle" if state_name == initial_state else "circle"
            g.node(state_name, label=label, shape=shape)

    # Add transitions
    for i in range(num_states):
        from_state = state_names[i]

        for inp in input_alphabet:
            to_state = state_names[np.random.randint(num_states)]

            if is_mealy:
                # For Mealy, add output to transition
                output = np.random.choice(output_alphabet)
                g.edge(from_state, to_state, label=f"{inp}/{output}")
            else:
                # For Moore, just input on transition
                g.edge(from_state, to_state, label=inp)

    # Render the machine
    st.graphviz_chart(g)

    # Ask the question
    st.write("**Question:** Is this a Moore or Mealy machine?")
    # Use radio buttons and check session state for previous answer if submitted
    submitted = st.session_state.get("answer_submitted", False)
    user_choice = st.session_state.get("user_answer", None)

    answer_options = ["Moore", "Mealy"]
    default_index = answer_options.index(user_choice) if user_choice else 0

    user_answer = st.radio(
        "Select your answer:",
        answer_options,
        index=default_index,
        key="identify_answer",
        horizontal=True,
        disabled=submitted # Disable after submission
    )

    correct_answer = "Mealy" if is_mealy else "Moore"

    if not submitted:
        if st.button("Submit Answer", key="identify_submit"):
            st.session_state["answer_submitted"] = True
            st.session_state["user_answer"] = user_answer
            st.session_state["correct_answer"] = correct_answer
            st.rerun() # Rerun to disable radio and show result

    if submitted:
        user_ans = st.session_state["user_answer"]
        correct_ans = st.session_state["correct_answer"]
        if user_ans == correct_ans:
            st.success(f"✔️ Correct! This is a **{correct_ans}** machine.")
            # Provide explanation
            if is_mealy:
                st.info("💡 **Reason:** In a Mealy machine, the outputs are associated with the *transitions* (shown as `input/output` on each arrow). The states themselves don't inherently have outputs.")
            else:
                st.info("💡 **Reason:** In a Moore machine, the outputs are associated with the *states* themselves (shown inside or alongside each state circle). Transition arrows only show the input symbol.")
        else:
            st.error(f"❌ Incorrect. You chose {user_ans}, but this is a **{correct_ans}** machine.")
            # Provide explanation
            if is_mealy:
                 st.info("💡 **Reason:** Look at the arrows - they are labeled `input/output`. This signifies a Mealy machine where output depends on the transition.")
            else:
                 st.info("💡 **Reason:** Look at the state circles - they contain an output value (e.g., 'A', 'B'). This signifies a Moore machine where output depends on the current state.")


def create_predict_output_problem(difficulty):
    st.subheader("Predict Output")
    st.write("Given the machine diagram and an input string, determine the resulting output sequence.")

    # Generate a random machine
    is_mealy = np.random.choice([True, False])

    if difficulty == "Easy":
        num_states = 2
        input_alphabet = ["0", "1"]
        test_length = 3
    elif difficulty == "Medium":
        num_states = 3
        input_alphabet = ["0", "1"]
        test_length = 4
    else:  # Hard
        num_states = 4
        input_alphabet = ["a", "b", "c"]
        test_length = 5

    output_alphabet = ["X", "Y"]
    if difficulty == "Hard":
        output_alphabet.append("Z")

    state_names = [f"q{i}" for i in range(num_states)]
    initial_state = state_names[0] # Always start at q0

    # Create the machine structures (transitions, outputs)
    transitions = {}
    transition_outputs = {} # For Mealy
    state_outputs = {} # For Moore

    for i in range(num_states):
        state_name = state_names[i]
        transitions[state_name] = {}
        if is_mealy:
            transition_outputs[state_name] = {}
        else:
            state_outputs[state_name] = np.random.choice(output_alphabet) # Assign Moore output

        for inp in input_alphabet:
            to_state = state_names[np.random.randint(num_states)]
            transitions[state_name][inp] = to_state
            if is_mealy:
                transition_outputs[state_name][inp] = np.random.choice(output_alphabet) # Assign Mealy output


    # Create the visualization
    g = graphviz.Digraph()
    g.attr(rankdir="LR")

    # Add states
    for i in range(num_states):
        state_name = state_names[i]
        shape = "doublecircle" if state_name == initial_state else "circle"
        if is_mealy:
            g.node(state_name, shape=shape)
        else:
            label = f"{state_name}\n{state_outputs[state_name]}"
            g.node(state_name, label=label, shape=shape)

    # Add transitions
    for from_state, trans_dict in transitions.items():
        for inp, to_state in trans_dict.items():
            if is_mealy:
                output = transition_outputs[from_state][inp]
                g.edge(from_state, to_state, label=f"{inp}/{output}")
            else:
                g.edge(from_state, to_state, label=inp)

    # Render the machine
    st.graphviz_chart(g)

    # Generate test input
    test_input = ''.join(np.random.choice(input_alphabet, size=test_length))

    # Calculate correct output (Carefully!)
    correct_output_sequence = []
    calculated_correct_output = ""
    current_state = initial_state
    valid_calculation = True
    trace_explanation = []

    if not is_mealy:
        # Moore: output for the initial state first
        if initial_state in state_outputs:
            initial_out = state_outputs[initial_state]
            correct_output_sequence.append(initial_out)
            trace_explanation.append(f"Start at **{initial_state}**, initial output: **{initial_out}**")
        else:
             # This should not happen with the generation logic, but good to check
            st.error("Error: Initial state output missing during calculation.")
            valid_calculation = False
    else:
        # Mealy: No initial output, starts with first transition
         trace_explanation.append(f"Start at **{initial_state}**")


    if valid_calculation:
        temp_state = current_state
        for i, symbol in enumerate(test_input):
            if temp_state not in transitions or symbol not in transitions[temp_state]:
                st.error(f"Error: Missing transition ({temp_state}, {symbol}) in generated machine.")
                valid_calculation = False
                break

            next_state = transitions[temp_state][symbol]

            if is_mealy:
                if temp_state not in transition_outputs or symbol not in transition_outputs[temp_state]:
                    st.error(f"Error: Missing output for transition ({temp_state}, {symbol}).")
                    valid_calculation = False
                    break
                output = transition_outputs[temp_state][symbol]
                correct_output_sequence.append(output)
                trace_explanation.append(f"Input '{symbol}': **{temp_state}** → **{next_state}**, output **{output}**")
            else:
                # Moore: output is determined by the *next* state
                if next_state not in state_outputs:
                    st.error(f"Error: Missing output for state {next_state}.")
                    valid_calculation = False
                    break
                output = state_outputs[next_state]
                correct_output_sequence.append(output)
                trace_explanation.append(f"Input '{symbol}': **{temp_state}** → **{next_state}**, state output: **{output}**")

            temp_state = next_state # Move to next state for the next iteration

        calculated_correct_output = "".join(correct_output_sequence)


    if not valid_calculation:
        st.error("Could not calculate the correct output due to an internal error in machine generation. Please try 'New Problem'.")
        return

    # Provide the machine type and test input
    machine_type = "Mealy" if is_mealy else "Moore"
    st.write(f"This is a **{machine_type}** machine.")
    st.write(f"**Input string:** `{test_input}`")

    # Ask for the output
    submitted = st.session_state.get("answer_submitted", False)
    user_input_val = st.session_state.get("user_output", "")

    user_output = st.text_input(
        "What is the output string?",
        value=user_input_val,
        key="predict_answer",
        disabled=submitted
    )

    if not submitted:
        if st.button("Submit Answer", key="predict_submit"):
            st.session_state["answer_submitted"] = True
            st.session_state["user_answer"] = user_output # Store user text input
            st.session_state["user_output"] = user_output # Keep value in text box
            st.session_state["correct_answer"] = calculated_correct_output
            st.session_state["trace_explanation"] = trace_explanation # Store trace
            st.rerun()

    if submitted:
        user_ans = st.session_state["user_answer"]
        correct_ans = st.session_state["correct_answer"]
        if user_ans == correct_ans:
            st.success(f"✔️ Correct! The output is: `{correct_ans}`")
        else:
            st.error(f"❌ Incorrect. Your answer: `{user_ans}`. The correct output is: `{correct_ans}`")

            # Explain step by step using the stored trace
            st.write("---")
            st.write("🧠 **Step-by-step explanation:**")
            trace = st.session_state.get("trace_explanation", [])
            for step in trace:
                st.write(step)


def create_complete_transitions_problem(difficulty):
    st.subheader("Complete Transitions")
    st.write("Fill in the missing transitions/outputs (marked as '❓') to satisfy the given input/output examples.")

    # Generate a random machine structure
    is_mealy = np.random.choice([True, False])

    if difficulty == "Easy":
        num_states = 2
        input_alphabet = ["0", "1"]
        missing_count = 1
        num_examples = 2
        example_length = 3
    elif difficulty == "Medium":
        num_states = 3
        input_alphabet = ["0", "1"]
        missing_count = 2
        num_examples = 2
        example_length = 4
    else:  # Hard
        num_states = 3 # Keep states manageable for completion
        input_alphabet = ["a", "b"]
        missing_count = 3
        num_examples = 3
        example_length = 5

    output_alphabet = ["X", "Y"]
    state_names = [f"q{i}" for i in range(num_states)]
    initial_state = state_names[0]

    # --- 1. Generate the COMPLETE target machine first ---
    target_transitions = {}
    target_transition_outputs = {} # For Mealy
    target_state_outputs = {} # For Moore

    for i in range(num_states):
        state_name = state_names[i]
        target_transitions[state_name] = {}
        if is_mealy:
            target_transition_outputs[state_name] = {}
        else:
            # Ensure all states have outputs for Moore
            target_state_outputs[state_name] = np.random.choice(output_alphabet)

        for inp in input_alphabet:
            target_transitions[state_name][inp] = state_names[np.random.randint(num_states)]
            if is_mealy:
                target_transition_outputs[state_name][inp] = np.random.choice(output_alphabet)

    # --- 2. Generate example input-output pairs using the COMPLETE machine ---
    examples = []
    expected_outputs = []
    attempts = 0
    max_attempts = 20 # Avoid infinite loop if generation is tricky
    while len(examples) < num_examples and attempts < max_attempts:
        attempts += 1
        test_input = ''.join(np.random.choice(input_alphabet, size=example_length))
        if test_input in examples: continue # Avoid duplicate inputs

        # Simulate the COMPLETE machine to get the correct output
        output_sequence = []
        current_state = initial_state
        valid_example = True

        if not is_mealy:
            if initial_state in target_state_outputs:
                 output_sequence.append(target_state_outputs[initial_state])
            else: # Should not happen
                 valid_example = False

        if valid_example:
             for symbol in test_input:
                if current_state not in target_transitions or symbol not in target_transitions[current_state]:
                     valid_example = False; break # Should not happen
                next_state = target_transitions[current_state][symbol]

                if is_mealy:
                    if current_state not in target_transition_outputs or symbol not in target_transition_outputs[current_state]:
                         valid_example = False; break # Should not happen
                    output = target_transition_outputs[current_state][symbol]
                    output_sequence.append(output)
                else: # Moore
                    if next_state not in target_state_outputs:
                         valid_example = False; break # Should not happen
                    output = target_state_outputs[next_state]
                    output_sequence.append(output)

                current_state = next_state

        if valid_example:
            examples.append(test_input)
            expected_outputs.append("".join(output_sequence))

    if len(examples) < num_examples:
        st.error("Failed to generate enough valid examples for the target machine. Try 'New Problem'.")
        return

    # --- 3. Select transitions to remove/hide ---
    all_possible_transitions = [(f"q{i}", inp) for i in range(num_states) for inp in input_alphabet]
    np.random.shuffle(all_possible_transitions) # Shuffle to pick random ones
    missing_transitions_info = [] # Store (from_state, input) tuple for missing ones

    # Ensure missing_count doesn't exceed available transitions
    actual_missing_count = min(missing_count, len(all_possible_transitions))

    for i in range(actual_missing_count):
         missing_transitions_info.append(all_possible_transitions[i])

    # Store the correct answers for the missing parts
    correct_missing_data = {} # Key: (from_state, input), Value: {'to': state, 'out': output (for Mealy)}
    for from_s, inp in missing_transitions_info:
         to_s = target_transitions[from_s][inp]
         correct_missing_data[(from_s, inp)] = {'to': to_s}
         if is_mealy:
             correct_missing_data[(from_s, inp)]['out'] = target_transition_outputs[from_s][inp]


    # --- 4. Create visualization of the INCOMPLETE machine ---
    g = graphviz.Digraph()
    g.attr(rankdir="LR")

    # Add states
    for i in range(num_states):
        state_name = state_names[i]
        shape = "doublecircle" if state_name == initial_state else "circle"
        if is_mealy:
            g.node(state_name, shape=shape)
        else:
            label = f"{state_name}\n{target_state_outputs[state_name]}" # Show outputs for Moore
            g.node(state_name, label=label, shape=shape)

    # Add transitions (only the ones NOT missing)
    for from_state, trans_dict in target_transitions.items():
        for inp, to_state in trans_dict.items():
            if (from_state, inp) not in missing_transitions_info: # If transition is NOT missing
                 if is_mealy:
                     output = target_transition_outputs[from_state][inp]
                     g.edge(from_state, to_state, label=f"{inp}/{output}")
                 else:
                     g.edge(from_state, to_state, label=inp)
            else: # If transition IS missing, draw a placeholder edge or node? Let's skip drawing it.
                 # Optionally add a visual cue, like a dashed edge placeholder if needed,
                 # but usually just showing the known ones is clearer.
                 pass


    # Render the incomplete machine
    st.write("### Incomplete Machine Diagram")
    st.graphviz_chart(g)

    # Display the required Input/Output behavior
    st.write("### Required Behavior")
    st.write("Your completed machine must produce the following outputs for these inputs:")
    for i in range(len(examples)):
        st.write(f"- Input: `{examples[i]}` → Expected Output: `{expected_outputs[i]}`")

    # --- 5. Create form for user to fill in the missing parts ---
    st.write("### Fill in the Missing Parts")
    st.write(f"Machine Type: **{'Mealy' if is_mealy else 'Moore'}**")

    user_inputs = {} # To store the widgets' current values

    # Sort missing transitions for consistent display order
    missing_transitions_info.sort()

    for from_state, inp in missing_transitions_info:
        st.write(f"From **{from_state}** on input **'{inp}'**: ")
        cols = st.columns(2 if is_mealy else 1)

        key_base = f"complete_{from_state}_{inp}"
        submitted = st.session_state.get("complete_check_result", None) is not None

        with cols[0]:
            # Retrieve previous selection if available and submitted==False? Let's reset on new problem.
            user_to_state = st.selectbox(
                f"Next State (δ({from_state},{inp}))",
                state_names,
                key=f"{key_base}_to",
                index=None, # Force user selection initially
                placeholder="Select state...",
                disabled=submitted
            )
            user_inputs[(from_state, inp)] = {'to': user_to_state}

        if is_mealy:
             with cols[1]:
                user_output = st.selectbox(
                    f"Output (λ({from_state},{inp}))",
                    output_alphabet,
                    key=f"{key_base}_out",
                    index=None,
                    placeholder="Select output...",
                    disabled=submitted
                )
                user_inputs[(from_state, inp)]['out'] = user_output

    # --- 6. Button to check the solution ---
    if not submitted:
        if st.button("Check My Solution", key="complete_check"):
            all_filled = True
            # Preliminary check: Did the user select something for every missing part?
            for key_info, selections in user_inputs.items():
                if selections['to'] is None:
                    all_filled = False
                if is_mealy and selections['out'] is None:
                    all_filled = False

            if not all_filled:
                st.warning("⚠️ Please fill in all missing transitions and outputs before checking.")
            else:
                # Construct the user's completed machine definition
                user_transitions = {state: dict(trans) for state, trans in target_transitions.items()}
                user_transition_outputs = {}
                if is_mealy:
                    user_transition_outputs = {state: dict(trans) for state, trans in target_transition_outputs.items()}

                # Overwrite with user's inputs for the missing parts
                for (from_s, inp), selections in user_inputs.items():
                    user_transitions[from_s][inp] = selections['to']
                    if is_mealy:
                        user_transition_outputs[from_s][inp] = selections['out']

                # Now, simulate the user's machine for each example
                results = []
                correct_count = 0
                for i in range(len(examples)):
                    test_input = examples[i]
                    expected_output = expected_outputs[i]
                    generated_output_sequence = []
                    current_state = initial_state
                    valid_run = True
                    run_trace = []

                    if not is_mealy:
                        if initial_state in target_state_outputs: # Use target_state_outputs as Moore outputs are fixed
                             gen_out = target_state_outputs[initial_state]
                             generated_output_sequence.append(gen_out)
                             run_trace.append(f"Start at **{initial_state}**, initial output: **{gen_out}**")
                        else:
                             valid_run = False; run_trace.append("Error: Missing Moore output for initial state.")
                    else:
                         run_trace.append(f"Start at **{initial_state}**")


                    if valid_run:
                        for symbol in test_input:
                             # Check if transition exists in the user's completed machine
                             if current_state not in user_transitions or symbol not in user_transitions[current_state]:
                                 valid_run = False; run_trace.append(f"Error: Transition from **{current_state}** on input **'{symbol}'** is undefined in your solution!"); break
                             next_state = user_transitions[current_state][symbol]

                             if is_mealy:
                                 # Check if output exists
                                 if current_state not in user_transition_outputs or symbol not in user_transition_outputs[current_state]:
                                     valid_run = False; run_trace.append(f"Error: Output for transition (**{current_state}**, **'{symbol}'**) is undefined in your solution!"); break
                                 gen_out = user_transition_outputs[current_state][symbol]
                                 generated_output_sequence.append(gen_out)
                                 run_trace.append(f"Input '{symbol}': **{current_state}** → **{next_state}**, output **{gen_out}**")
                             else: # Moore
                                 # Check if output exists for the destination state
                                 if next_state not in target_state_outputs:
                                     valid_run = False; run_trace.append(f"Error: Moore output for state **{next_state}** is undefined!"); break
                                 gen_out = target_state_outputs[next_state]
                                 generated_output_sequence.append(gen_out)
                                 run_trace.append(f"Input '{symbol}': **{current_state}** → **{next_state}**, state output: **{gen_out}**")

                             current_state = next_state


                    final_generated_output = "".join(generated_output_sequence)
                    is_correct = valid_run and (final_generated_output == expected_output)
                    if is_correct:
                         correct_count += 1

                    results.append({
                        "input": test_input,
                        "expected": expected_output,
                        "generated": final_generated_output if valid_run else "ERROR",
                        "correct": is_correct,
                        "trace": run_trace,
                        "valid_run": valid_run
                    })

                # Store results in session state
                st.session_state["complete_check_result"] = {
                    "total_correct": correct_count,
                    "total_examples": len(examples),
                    "details": results,
                    "user_inputs": user_inputs, # Store what user selected
                    "correct_missing": correct_missing_data # Store the actual solution
                }
                st.rerun() # Rerun to show results and disable inputs

    # --- 7. Display results after checking ---
    if st.session_state.get("complete_check_result", None) is not None:
        result_data = st.session_state["complete_check_result"]
        total_correct = result_data["total_correct"]
        total_examples = result_data["total_examples"]
        details = result_data["details"]
        user_sel = result_data["user_inputs"]
        correct_sel = result_data["correct_missing"]

        st.divider()
        if total_correct == total_examples:
             st.success(f"🎉 **Excellent!** Your completed machine correctly satisfies all {total_examples} examples.")
        else:
             st.error(f" Mmm, **Not quite right.** Your machine satisfied {total_correct} out of {total_examples} examples.")

        st.write("---")
        st.write("#### Detailed Results:")

        for i, res in enumerate(details):
             expander_title = f"Example {i+1}: Input=`{res['input']}` → Expected=`{res['expected']}`"
             if res['correct']:
                 expander_title += " ✔️"
             else:
                  expander_title += f" ❌ (Your Output: `{res['generated']}`)"

             with st.expander(expander_title):
                st.write("**Execution Trace (Your Machine):**")
                if not res["valid_run"]:
                    st.warning("Simulation stopped due to an error or undefined transition/output in your solution.")
                for step in res["trace"]:
                     st.write(f"- {step}")
                if not res['correct'] and res['valid_run']:
                    st.write(f"Comparison: Expected `{res['expected']}`, but got `{res['generated']}`")

        # Optionally show the correct solution for the missing parts if incorrect
        if total_correct < total_examples:
             st.write("---")
             st.write("💡 **Hint: Correct Missing Parts**")
             for (from_s, inp), correct_data in correct_sel.items():
                 user_to = user_sel.get((from_s, inp), {}).get('to', 'Not Set')
                 correct_to = correct_data['to']
                 msg = f"- Transition (**{from_s}**, **'{inp}'**): Correct Next State is **{correct_to}**"
                 if user_to != correct_to:
                     msg += f" (You chose: {user_to})"

                 if is_mealy:
                     user_out = user_sel.get((from_s, inp), {}).get('out', 'Not Set')
                     correct_out = correct_data['out']
                     msg += f", Correct Output is **{correct_out}**"
                     if user_out != correct_out:
                        msg += f" (You chose: {user_out})"
                 st.write(msg)


# --- Main execution ---
if __name__ == "__main__":
    main()