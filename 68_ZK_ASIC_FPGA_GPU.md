# ETAAcademy-ZKMeme: 68. ZK ASIC, FPGA & GPU

<table>
  <tr>
    <th>title</th>
    <th>tags</th>
  </tr>
  <tr>
    <td>68. ZK ASIC, FPGA & GPU</td>
    <td>
      <table>
        <tr>
          <th>zk-meme</th>
          <th>basic</th>
          <th>quick_read</th>
          <td>ZK_ASIC_FPGA_GPU</td>
        </tr>
      </table>
    </td>
  </tr>
</table>

[Github](https://github.com/ETAAcademy)｜[Twitter](https://twitter.com/ETAAcademy)｜[ETA-ZK-Meme](https://github.com/ETAAcademy/ETAAcademy-ZK-Meme)

Authors: [Evta](https://twitter.com/pwhattie), looking forward to your joining

# Bridging Algorithms and Circuits: Hardware Design for ZK Proofs

Zero-knowledge (ZK) proof hardware acceleration spans a full spectrum from general-purpose processors to fully specialized chips. **CPUs** are highly flexible but constrained by control logic overhead, which makes large-scale parallelism impractical. **GPUs**, though massively parallel, are optimized for floating-point computation and struggle with large-integer arithmetic and irregular memory access patterns. **FPGAs** enable direct implementation of custom circuits in HDL, offering up to an order-of-magnitude improvement in energy efficiency over GPUs at lower cost, though they demand specialized hardware expertise. At the far end of the spectrum, **ASICs** provide the highest performance and lowest power consumption, with the potential to shrink proof times to just a few seconds. However, they require 1–2 years of development and multi-million-dollar investment, making them viable only for large-scale, mature applications.

At the implementation level, **hardware description languages (HDLs)** such as Verilog and VHDL are central to both ASIC and FPGA development. They bridge algorithmic and hardware design thinking, offering rich operators, data types, and control constructs to capture circuit behavior precisely—while also forcing developers to confront engineering challenges such as simulation–synthesis mismatches. The entire design stack rests on **digital circuit fundamentals**: combinational logic (e.g., adders, decoders, multiplexers) delivers immediate computation, while sequential logic (e.g., flip-flops, registers) maintains state and memory, together forming the foundation of modern computer architecture. The ASIC design flow proceeds from RTL coding through simulation, synthesis, and place-and-route to final fabrication, increasingly aided by open-source toolchains such as OpenLANE2 and community platforms like TinyTapeout, with an emphasis on IP reuse and modular development.

Meanwhile, **FPGAs**, as “fixed-architecture but reconfigurable” platforms, map HDL logic onto limited hardware primitives (LUTs, flip-flops, BRAMs, DSPs), requiring resource-aware design to fully exploit available capacity. **GPUs**, supported by CUDA-based toolchains, already offer relatively mature ZK acceleration solutions. Yet their SIMT execution model and hierarchical memory system demand careful optimization—through techniques such as memory coalescing, reducing branch divergence, and exploiting asynchronous streams—to unlock high throughput.

---

### Hardware Acceleration for Zero-Knowledge Proofs: From CPU, FRGA to ASIC

The core of zero-knowledge (ZK) hardware acceleration lies in **parallelizing key operations** such as multi-scalar multiplication (MSM) and number-theoretic transforms (NTT). Different hardware architectures offer distinct trade-offs in flexibility, performance, and efficiency, forming a continuum from general-purpose CPUs to highly specialized ASICs.

**CPU: Flexibility with Limited Parallelism**
CPUs are designed for versatility, dedicating much of their silicon area to control logic and caching rather than arithmetic units. While this architecture makes CPUs highly flexible, it also limits their parallel computation capacity. As a result, CPUs struggle to provide the throughput required for large-scale ZK proofs.

**GPU: Massive Parallelism with Bottlenecks**
GPUs allocate a much larger proportion of chip area to arithmetic units, making them well-suited for massive parallelism. Leveraging CUDA SDKs (such as NVIDIA CUDA), developers can accelerate MSM and NTT by writing code almost as if it were native, without delving into hardware details. Despite their computational power, GPUs are optimized primarily for floating-point operations. Moreover, their memory hierarchy creates bottlenecks in ZK workloads: each thread has access to only limited shared memory, and irregular global memory access patterns significantly degrade performance for polynomial and MSM computations.

**FPGA: Custom Logic and High Energy Efficiency**
Field-Programmable Gate Arrays (FPGAs) enable developers to implement algorithms directly at the circuit level using hardware description languages (HDLs). Unlike GPUs, they require no instruction decoding or compilation, providing greater customization and efficiency. FPGAs are typically one-third the cost of GPUs while delivering more than ten times the energy efficiency. By adding dedicated computation modules, FPGAs can handle MSM and NTT workloads with minimal power overhead, making them ideal for compute-intensive, high-throughput, low-latency ZK proof scenarios. However, a major barrier is talent scarcity: it is challenging to assemble teams with expertise in both cryptography and FPGA engineering.

**ASIC: The Ultimate Specialized Solution**
Application-Specific Integrated Circuits (ASICs) hardwire algorithms directly into silicon, offering unmatched performance and minimal power consumption. Like FPGAs, they provide hardware-level acceleration, but as fixed-function circuits, they can be further optimized for ZK workloads such as large integer arithmetic and irregular memory access patterns. This makes ASICs the most promising path toward reducing ZK proof generation to seconds. Yet, ASIC development comes with high barriers: design and production require 1–2 years and an investment of \$10–20 million. Given the risk of rapid protocol evolution, ASICs are only viable once ZK technology reaches sufficient stability to avoid obsolescence.

---

## 1. Hardware Description Languages & Digital Circuit Design for ASICs

Application-Specific Integrated Circuits (ASICs) are custom-built chips designed for specialized tasks, offering the highest performance and energy efficiency in domains such as zero-knowledge (ZK) proofs and AI acceleration. At the heart of ASIC design—and particularly relevant for ZK workloads—are **Hardware Description Languages (HDLs)** and **digital circuit design**. Commonly, Verilog or VHDL is used to describe logic functions at the **Register Transfer Level (RTL)**, providing a higher level of abstraction than transistor-level design and enabling modularity and reuse. To progress from simple components to complex systems, one must first understand the fundamentals: logic gates, flip-flops, combinational circuits, and sequential logic.

**From RTL to Simulation**
RTL designs are verified using simulators such as **Verilator** and **Icarus Verilog**, where testbenches drive the design and waveform viewers (Surfer, VaporView, GTKWave) visualize signal behavior over time. This simulation stage helps bridge the gap between code and hardware behavior, building intuition for how logic translates into real circuits.

**From RTL to GDSII**
To manufacture chips, RTL descriptions must eventually be transformed into a physical layout via the **RTL-to-GDSII toolchain**. Tools such as **OpenROAD** provide full-flow support but rely heavily on TCL scripting. **OpenLANE/OpenLANE2** introduce more accessible Python- and JSON-based configuration, with OpenLANE2 in particular favored for its centralized configuration and debugging-friendly workflow. Another option, **SiliconCompiler**, provides flexible build management. Across the board, modern ASIC design emphasizes **reuse and integration**, leveraging open-source IP cores rather than developing every module from scratch. For instance, RISC-V processor cores exist across a spectrum: from **SERV**, an ultra-minimal 1-bit serialized core, to **PicoRV32**, a lightweight multi-cycle core, to **Ibex**, a higher-performance pipelined core. Tools like **Chipyard** allow designers to integrate CPUs, memory, and peripherals into complete System-on-Chip (SoC) architectures, extending the flow from single RTL modules to full SoCs.

**Verification and Testing**
Because ASIC fabrication is expensive, rigorous verification is non-negotiable. Beyond traditional Verilog testbenches, **CocoTB**—a Python-based verification framework—enables developers to write more expressive, high-level tests. Meanwhile, tools like **SymbiYosys** bring formal verification into the flow, offering mathematical assurance of correctness.

**Development Environments and Workflow**
Installing and managing ASIC design toolchains is notoriously complex. To simplify this, containerized environments such as **IIC-OSIC-TOOLS (Docker)** package simulators, EDA tools, and verification frameworks into a unified, cross-platform environment. Paired with VS Code integration, developers can perform coding, simulation, and waveform analysis seamlessly in a single workspace. To lower the learning curve of command-line-heavy flows, **Makefiles** are often used to wrap complex commands into simple targets (e.g., `make lint`, `make tests`, `make openlane`), while still allowing advanced users to access the underlying commands directly.

**From High Costs to Accessible Fabrication**
Whereas ASIC tape-outs once cost hundreds of thousands of dollars, recent initiatives have dramatically reduced barriers. **Multi-Project Wafers (MPW)** and platforms such as **TinyTapeout** now enable small-scale fabrication for only a few hundred dollars. This democratization of ASIC production allows individuals and small teams to prototype custom chips, opening the door for broader experimentation in fields like ZK acceleration.

---

### 1.1 Fundamentals of Hardware Description Languages (HDLs)

A **Hardware Description Language (HDL)** is a specialized programming language used to describe the structure and behavior of digital circuits. Unlike software programming, which models sequential execution, HDLs capture **concurrent hardware behavior**. The most widely used languages include **Verilog**, which has C-like syntax and is favored in industry for ASIC design; **VHDL**, which emphasizes strong typing and structured design, making it ideal for safety-critical applications; and **SystemVerilog**, an extension of Verilog that introduces object-oriented features and advanced verification capabilities, now the de facto standard for complex SoC projects.

In modern ASIC and FPGA workflows, HDLs are far more than code—they are the primary medium through which designers **express intent and organize complex systems**. For example, Verilog supports **multi-bit logic variables** declared as `logic [N-1:0] var_name;`. Constants can be written in multiple formats: binary (`4'b0101`), hexadecimal (`12'hCA5`), or decimal (`12'd1058`).

**Operators** form the foundation of digital logic design. Bitwise operators (`&`, `|`, `^`, `~`) map directly to logic gates and operate on vectors. Reduction operators collapse multi-bit signals into single-bit results, useful for parity checks or zero detection. Arithmetic operators (`+`, `-`) describe adders and subtractors, though designers must account for wraparound behavior on overflow. Shift operators (`<<`, `>>`, `>>>`) correspond to hardware shifters, with logical shifts filling with zeros and arithmetic shifts preserving the sign bit. Relational operators (`==`, `!=`, `<`, `>`) generate comparators, though equality checks handle unknown values (`X`) differently than ordering comparisons. The **concatenation operator** (`{}`) is a powerful Verilog feature, enabling designers to merge small bit vectors into larger buses, while the replication form `{N{var}}` efficiently duplicates patterns. These operators not only provide rich logical expressiveness but also **map directly to hardware structures**, bridging algorithmic thinking and hardware realization.

<details><summary>Code</summary>

```Verilog

/*
     * Verilog Basic Operators and Data Types Testbench
     * Demonstrates the usage and behavior of various operators
     */
    module verilog_operators_testbench;

      /*
       * 1. Single-bit logic data type
       * - Basic hardware signal representation
       * - Supports four logic values: 0, 1, X, Z
       * - X: unknown value, Z: high impedance
       */
      logic a, b, c;

      /*
       * 2. Multi-bit logic data type
       * - [MSB:LSB] format: [3:0] represents 4 bits
       * - Bit indexing starts from 0 (LSB is 0)
       * - Supports multiple radix literals
       */
      logic [3:0] A;  // 4-bit vector
      logic [3:0] B;  // 4-bit vector
      logic [3:0] C;  // 4-bit vector
      logic [7:0] D;  // 8-bit vector
      logic [11:0] E; // 12-bit vector

      initial begin
        $display("=== Verilog Operators and Data Types Test ===");

        // Single-bit literal testing
        $display("\n1. Single-bit Logic Data Type Test:");
        a = 1'b0;  $display("a = 1'b0 = %b", a);
        a = 1'b1;  $display("a = 1'b1 = %b", a);
        a = 1'bx;  $display("a = 1'bx = %b", a);
        a = 1'bz;  $display("a = 1'bz = %b", a);

        // Bitwise logic operator testing
        $display("\nBitwise Logic Operator Test:");
        a = 1'b0; b = 1'b1;
        c = a & b;  $display("0 & 1 = %b", c);
        c = a | b;  $display("0 | 1 = %b", c);
        c = a ^ b;  $display("0 ^ 1 = %b", c);
        c = ~b;     $display("~1 = %b", c);

        // Multi-bit literal format testing
        $display("\n2. Multi-bit Logic Data Type Test:");
        A = 4'b0101;           $display("A = 4'b0101 = %b", A);
        D = 12'hca5;           $display("D = 12'hca5 = %h", D);
        D = 12'd1058;          $display("D = 12'd1058 = %d", D);
        D = 12'b1100_1010_0101; $display("D = 12'b1100_1010_0101 = %b", D);

        /*
         * 3. Bitwise operators - bit-by-bit operations
         * - &, |, ^, ~: perform operations on each bit separately
         * - Input and output bit widths are the same
         */
        $display("\n3. Bitwise Operator Test:");
        A = 4'b0101; B = 4'b0011;
        C = A & B;  $display("4'b0101 & 4'b0011 = %b", C);
        C = A | B;  $display("4'b0101 | 4'b0011 = %b", C);
        C = A ^ B;  $display("4'b0101 ^ 4'b0011 = %b", C);

        /*
         * Boolean operators - overall logical evaluation
         * - &&, ||, !: treat operands as overall true/false values
         * - Output is always 1 bit
         */
        $display("\nBoolean Operator Test:");
        a = 1'b0; b = 1'b1;
        c = a && b; $display("0 && 1 = %b", c);
        c = a || b; $display("0 || 1 = %b", c);
        c = !b;     $display("!1 = %b", c);

        /*
         * 4. Reduction operators - multi-bit reduced to single bit
         * - Reduce all bits of a multi-bit vector to a single bit result
         * - &A: AND all bits | |A: OR all bits | ^A: XOR all bits
         */
        $display("\n4. Reduction Operator Test:");
        A = 4'b0101;
        C = &A;   $display("&(4'b0101) = %b (AND all bits)", C);
        C = |A;   $display("|(4'b0101) = %b (OR all bits)", C);
        C = ^A;   $display("^(4'b0101) = %b (even parity)", C);

        /*
         * 5. Shift operators
         * - <<, >>: logical left/right shift (zero fill)
         * - >>>: arithmetic right shift (sign extension)
         * - Supports both fixed and variable shift amounts
         */
        $display("\n5. Shift Operator Test:");
        A = 8'b1110_0101;
        $display("Original value: A = %b", A);

        // Logical shifts
        C = A << 2;  $display("A << 2 = %b (left shift 2 bits, zero fill)", C);
        C = A >> 2;  $display("A >> 2 = %b (right shift 2 bits, zero fill)", C);

        // Arithmetic right shift (requires $signed conversion)
        C = $signed(A) >>> 2;  $display("$signed(A) >>> 2 = %b (sign extension)", C);

        /*
         * 6. Arithmetic operators - unsigned integer operations
         * - +, -: addition and subtraction
         * - Note overflow and underflow wraparound behavior
         * - Avoid using *, /, %, ** (not synthesizable)
         */
        $display("\n6. Arithmetic Operator Test:");
        A = 8'd250;  // 250
        B = 8'd15;   // 15
        $display("A = %d, B = %d", A, B);

        C = A + B;   $display("A + B = %d (250 + 15, overflow wraparound)", C);
        C = B - A;   $display("B - A = %d (15 - 250, underflow wraparound)", C);

        /*
         * 7. Relational operators - comparison operations
         * - ==, !=: equality and inequality comparison
         * - >, <, >=, <=: magnitude comparison
         * - X value handling: == and != are smarter, magnitude comparison returns X
         */
        $display("\n7. Relational Operator Test:");
        A = 4'b1100;
        B = 4'b10xx;  // Contains unknown bits
        $display("A = %b, B = %b", A, B);

        a = (A == B);  $display("(A == B) = %b (cannot determine equality)", a);
        a = (A != B);  $display("(A != B) = %b (known bits differ, definitely unequal)",
  a);
        a = (A > B);   $display("(A > B) = %b (magnitude comparison with X returns X)",
  a);

        // Signed comparison
        $display("\nSigned Comparison Test:");
        A = 4'b1111;  // unsigned 15, signed -1
        B = 4'b0001;  // 1
        $display("A = %b (%d unsigned, %d signed)", A, A, $signed(A));
        $display("B = %b (%d)", B, B);
        a = (A > B);             $display("A > B (unsigned) = %b", a);
        a = ($signed(A) > $signed(B)); $display("$signed(A) > $signed(B) = %b", a);

        /*
         * 8. Concatenation operators - bit vector concatenation
         * - {}: basic concatenation operator
         * - {n{}}: replication operator
         * - Arranged left to right (left side in MSB)
         */
        $display("\n8. Concatenation Operator Test:");
        A = 4'ha;  // 1010
        B = 4'hb;  // 1011
        C = 4'hc;  // 1100
        $display("A = %h, B = %h, C = %h", A, B, C);

        // Basic concatenation
        E = {A, B, C};  $display("{A, B, C} = %h", E);
        E = {C, A, B};  $display("{C, A, B} = %h", E);

        // Replication operator
        E = {3{A}};     $display("{3{A}} = %h ({A, A, A})", E);
        E = {A, {2{B}}}; $display("{A, {2{B}}} = %h ({A, B, B})", E);

        $display("\n=== Test Complete ===");
        $finish;
      end

    endmodule

```

</details>

HDLs also offer higher-level abstractions for clarity and maintainability. **Enumerated types (enum)** improve code readability and safety by explicitly defining states, commonly used in finite state machines (FSMs). **Structures (struct)** provide a convenient way to model composite data such as RGB pixel values or 3D coordinates. With the `packed` keyword, structures map directly to contiguous bit vectors, easing the conversion between high-level fields and low-level signals. This eliminates error-prone manual concatenation and makes the code more semantically meaningful.

**Control-flow modeling** relies heavily on constructs such as the ternary operator, `if` statements, and `case` statements. The ternary operator is especially useful for compact multiplexer definitions but introduces the risk of **X-optimism**: when a condition evaluates to an unknown (`X`), simulation may produce results inconsistent with synthesized hardware. Similarly, careless use of `if` and `case` statements can create mismatches between simulation and synthesis. To avoid unintended latches and ensure safe synthesis, designers must always include a `default` branch in `case` statements.

Finally, **`casez`** is particularly valuable for priority encoders and leading-one detectors. It supports wildcard matching with `?`, offering flexibility while avoiding the pitfalls of **`casex`**, which incorrectly treats unknown values (`X`) as wildcards. Because of this, `casez` is considered safe for synthesis, whereas `casex` is strongly discouraged in professional design flows.

In short, HDLs provide the essential bridge from **algorithmic intent** to **hardware implementation**. Their constructs map directly to physical circuits, allowing designers to think both like programmers and hardware engineers—an indispensable skill for ASIC and FPGA development.

<details><summary>Code</summary>

```Verilog
/*
     * Verilog Advanced Features Testbench
     * Demonstrates the usage of enumerated types, structures, and control flow
  statements
     */
    module Verilog_advanced_testbench;

      /*
       * 1. enum - Enumerated type definition
       * - typedef: define new type
       * - logic [$clog2(4)-1:0]: base storage type (2-bit logic vector)
       * - $clog2(4): calculate required bits for storage
       * - _t suffix: distinguish type names from variable names
       */
      typedef enum logic [$clog2(4)-1:0] {
          STATE_A,
          STATE_B,
          STATE_C,
          STATE_D
      } state_t;

      // Additional state machine enumeration for demonstration
      typedef enum logic [1:0] {
          IDLE  = 2'b00,
          CALC  = 2'b01,
          DONE  = 2'b10
      } fsm_state_t;

      /*
       * 2. struct - Packed structure definition
       * - packed: ensures equivalent logic storage
       * - Bit field order: first field in MSB
       * - Bit layout: x[11:8] y[7:4] z[3:0]
       * Bit position: 11  10   9   8   7   6   5   4   3   2   1   0
       *              +---+---+---+---+---+---+---+---+---+---+---+---+
       * Fields:      |      x field    |      y field    |      z field    |
       *              |x[3]x[2]x[1]x[0] |y[3]y[2]y[1]y[0] |z[3]z[2]z[1]z[0]|
       *              +---+---+---+---+---+---+---+---+---+---+---+---+
       * Range:       [11:8]            [7:4]             [3:0]
       */
      typedef struct packed {
          logic [3:0] x;  // x field: bits[11:8]
          logic [3:0] y;  // y field: bits[7:4]
          logic [3:0] z;  // z field: bits[3:0]
      } point_t;

      // Signal declarations
      state_t current_state, next_state;
      fsm_state_t fsm_current, fsm_next;
      point_t point_a, point_b;
      logic [11:0] bits;
      logic [7:0] a, b, c;
      logic [1:0] sel;
      logic [3:0] test_vector;
      logic start, done, ack;

      initial begin
        $display("=== Verilog Advanced Features Test ===");

        /*
         * Enumerated type testing
         */
        $display("\n1. Enumerated Type Test:");
        current_state = STATE_A;
        $display("current_state = STATE_A, encoded value = %b", current_state);

        current_state = STATE_B;
        $display("current_state = STATE_B, encoded value = %b", current_state);

        current_state = STATE_C;
        $display("current_state = STATE_C, encoded value = %b", current_state);

        /*
         * State machine application demonstration - FSM state transitions
         * - current_state: current state register
         * - next_state: next state combinational logic
         * - case statement: state transition condition evaluation
         */
        $display("\nState Machine Transition Test:");
        fsm_current = IDLE;
        start = 1'b0; done = 1'b0; ack = 1'b0;

        // Simulate state transitions
        case (fsm_current)
            IDLE: fsm_next = start ? CALC : IDLE;
            CALC: fsm_next = done ? DONE : CALC;
            DONE: fsm_next = ack ? IDLE : DONE;
            default: fsm_next = IDLE;
        endcase
        $display("IDLE -> start=0 -> next_state = %s", fsm_next.name());

        start = 1'b1;
        case (fsm_current)
            IDLE: fsm_next = start ? CALC : IDLE;
            CALC: fsm_next = done ? DONE : CALC;
            DONE: fsm_next = ack ? IDLE : DONE;
            default: fsm_next = IDLE;
        endcase
        $display("IDLE -> start=1 -> next_state = %s", fsm_next.name());

        /*
         * Structure testing
         */
        $display("\n2. Structure Test:");

        /*
         * Structure operations - field access and type conversion
         * - Dot operator: access structure fields
         * - Whole assignment: direct assignment between structures
         * - Type conversion: conversion between structures and bit vectors
         */
        point_a.x = 4'h3;           // Field access
        point_a.y = 4'h5;
        point_a.z = 4'h7;
        $display("point_a: x=%h, y=%h, z=%h", point_a.x, point_a.y, point_a.z);

        point_b = point_a;          // Whole assignment
        $display("point_b (copied from point_a): x=%h, y=%h, z=%h", point_b.x, point_b.y,
   point_b.z);

        bits = point_a;             // Type conversion: structure to bit vector
        $display("point_a as bit vector: %h (12'h357)", bits);

        bits = 12'habc;
        point_a = bits;             // Type conversion: bit vector to structure
        $display("point_a restored from 12'habc: x=%h, y=%h, z=%h", point_a.x, point_a.y,
   point_a.z);

        /*
         * 3. Ternary operator test - conditional selection expression
         * - Syntax: condition ? true_value : false_value
         * - Nesting: supports multi-level conditional evaluation
         * - X value handling: definite bits remain definite, indefinite bits become X
         */
        $display("\n3. Ternary Operator Test:");

        // Basic usage
        a = 8'd10; b = 8'd20;
        c = (a < b) ? 8'd15 : 8'd14;
        $display("(10 < 20) ? 15 : 14 = %d", c);

        // Nested usage (multi-way selection)
        sel = 2'b01;
        c = (sel == 2'b00) ? 8'h0a :
            (sel == 2'b01) ? 8'h0b :
            (sel == 2'b10) ? 8'h0c :
            (sel == 2'b11) ? 8'h0d : 8'h0e;
        $display("Multi-way selection result with sel=01: %h", c);

        /*
         * 4. if statement X optimism test - simulation characteristic
         * - X values are treated as false
         * - May cause simulation/synthesis mismatch
         * - Need to be aware of X value effects
         */
        $display("\n4. if Statement X Optimism Test:");

        sel = 1'bx;
        if (sel == 1'b0) begin
            a = 8'h0a;
        end else begin
            a = 8'h0b;  // X treated as false
        end
        $display("if statement result when sel=X: a = %h (X optimism)", a);

        /*
         * 5. case statement test - multi-way branch selection
         * - Exact match: sel value must match completely
         * - default branch: handles unmatched cases
         * - X value handling: jumps to default branch, causes X optimism
         */
        $display("\n5. case Statement Test:");

        sel = 2'b01;
        case (sel)
            2'b00: a = 8'h0a;
            2'b01: a = 8'h0b;
            2'b10: a = 8'h0c;
            2'b11: a = 8'h0d;
            default: a = 8'h0e;  // Don't use X in case items, as hardware cannot match
  unknown values
        endcase
        $display("case selection result with sel=01: a = %h", a);

        // X value test
        sel = 2'bxx;
        case (sel)
            2'b00: a = 8'h0a;
            2'b01: a = 8'h0b;
            2'b10: a = 8'h0c;
            2'b11: a = 8'h0d;
            default: a = 8'h0e;
        endcase
        $display("case selection result with sel=XX: a = %h (jumps to default)", a);

        /*
         * 6. casez statement test - wildcard matching
         * - ? wildcard: matches 0 or 1
         * - Priority encoding: matches in order
         * - Applications: leading-one detector, priority encoder
         */
        $display("\n6. casez Statement Test:");

        test_vector = 4'b0100;
        casez (test_vector)
            4'b0000: b = 8'd0;
            4'b???1: b = 8'd1;  // ? is wildcard
            4'b??10: b = 8'd2;  // Priority encoding
            4'b?100: b = 8'd3;
            4'b1000: b = 8'd4;
            default: b = 8'hxx;
        endcase
        $display("casez result with test_vector=0100: b = %d (matches ?100)", b);

        test_vector = 4'b0001;
        casez (test_vector)
            4'b0000: b = 8'd0;
            4'b???1: b = 8'd1;  // Matches any ending with 1
            4'b??10: b = 8'd2;
            4'b?100: b = 8'd3;
            4'b1000: b = 8'd4;
            default: b = 8'hxx;
        endcase
        $display("casez result with test_vector=0001: b = %d (matches ???1)", b);

        test_vector = 4'b1010;
        casez (test_vector)
            4'b0000: b = 8'd0;
            4'b???1: b = 8'd1;
            4'b??10: b = 8'd2;  // Matches any ending with 10
            4'b?100: b = 8'd3;
            4'b1000: b = 8'd4;
            default: b = 8'hxx;
        endcase
        $display("casez result with test_vector=1010: b = %d (matches ??10)", b);

        $display("\n=== Verilog Advanced Features Test Complete ===");
        $finish;
      end

    endmodule
```

## </details>

### 1.2 Digital Circuits

Digital circuits are generally divided into two categories: **combinational circuits** and **sequential circuits**.

- **Combinational circuits** produce outputs that depend only on the **current inputs**, with no memory of past values. Examples include adders, multiplexers (MUX), encoders, and decoders. They are constructed purely from logic gates, without storage elements or feedback loops, and their outputs respond immediately to changes in input. In essence, combinational circuits handle **instantaneous computation and logical decision-making** within digital systems.

- **Sequential circuits**, by contrast, depend on both the **current inputs and historical states**. They incorporate memory elements, such as flip-flops or registers, and typically operate under the control of a clock signal. This gives them the ability to **store information and control flow**, making them essential for building counters, registers, and finite state machines.

---

#### Combinational Circuits

A **combinational circuit** is one where outputs are determined solely by the current set of inputs. Since no memory elements are present, these circuits lack feedback and exhibit no state. Instead, their functionality is expressed directly by Boolean logic, typically implemented using gates such as AND, OR, NOT, NAND, NOR, and XOR.

They are widely used in:

- **Arithmetic operations**: adders, subtractors, multipliers.
- **Data processing**: encoders, decoders, comparators, code converters.
- **Control logic**: multiplexers (MUX) for routing and selection.

Common building blocks include:

- **Half-Adder (HA)**: Computes the sum of two single-bit binary inputs. It produces a **Sum (S = A ⊕ B)** via XOR and a **Carry (C = A · B)** via AND. While simple, it cannot handle a carry-in, so it serves mainly as a building block for larger adders.

- **Full-Adder (FA)**: Extends the half-adder to include a carry-in input (Cin). It produces outputs:

  - **Sum (S) = A ⊕ B ⊕ Cin**
  - **Carry-out (Cout) = (A · B) + (Cin · (A ⊕ B))**
    A full-adder can be constructed from two half-adders plus an OR gate, and serves as the foundation for multi-bit adders such as ripple-carry and carry-lookahead adders.

- **Decoder**: Converts binary inputs into a unique activated output line. An **N-to-2ᴺ decoder** has N inputs and 2ᴺ outputs, with only one active at a time. For example, a 2-to-4 decoder activates one of four outputs based on the binary input. Decoders are widely used in **memory addressing, instruction decoding, and device selection**.

- **Encoder**: Performs the inverse of a decoder, converting multiple active inputs into a binary code. For instance, a 4-to-2 encoder generates a 2-bit code based on which input is active. **Priority encoders** resolve cases where multiple inputs are high by outputting the code of the highest-priority signal. Encoders are useful in **data compression, interrupt handling, and communication systems**.

- **Multiplexer (MUX)**: A “data selector” that routes one of several inputs to a single output, controlled by select lines. For example, a 2-to-1 MUX selects between two inputs based on a single control bit, while a 4-to-1 MUX uses two control bits to select among four inputs. MUXs are essential for **data routing, CPU datapaths, and memory addressing**.

- **Demultiplexer (DEMUX)**: The functional opposite of a MUX. A DEMUX takes a single input and routes it to one of many outputs, controlled by select lines. For instance, a 1-to-4 DEMUX directs its input to one of four outputs. DEMUXs are frequently used in **signal distribution, device selection, and communication buses**.

Overall, combinational circuits implement the concept **Output = f(Input)** with no stored state, making them fundamental building blocks for arithmetic units, CPU datapaths, and control logic.

---

#### Sequential Circuits

Unlike combinational circuits, **sequential circuits** have memory. Their outputs depend not only on the current input but also on the system’s **past state**. This state is stored in memory elements such as flip-flops or latches and is updated according to a clock signal. A sequential circuit typically includes:

- **Combinational logic** to compute the next state.
- **Memory elements** (flip-flops, registers) to store the current state.
- **Clock signals** to synchronize updates.
- **Feedback paths** from outputs to inputs, enabling state retention.

**Flip-Flops (FFs)** are the fundamental memory elements:

- **SR Latch**: Built from cross-coupled NAND gates, with inputs Set (S) and Reset (R). It can store a single bit but suffers from an undefined state when S=R=1. While simple, it is often extended to more stable flip-flops.

- **D Flip-Flop**: The most widely used storage element. It captures the value of the data input (D) on a clock edge and holds it until the next triggering event. This avoids the indeterminate states of SR latches. Applications include **registers, state machines, and synchronization circuits**.

- **JK Flip-Flop**: An improvement over the SR latch that eliminates invalid states. It accepts two inputs (J, K) and toggles its output when both are high. JK flip-flops are versatile, supporting set, reset, hold, and toggle operations. They are used in **counters, frequency dividers, and shift registers**.

- **T Flip-Flop**: A simplified version of the JK flip-flop that toggles state when T=1 on a clock edge. It is especially useful for **counters and clock division**.

From these primitives, more advanced sequential circuits are built:

- **Registers**: Collections of flip-flops storing multi-bit words.
- **Counters**: Sequential circuits that count clock pulses, often used in timers and dividers.
- **Shift Registers**: Move stored data left or right, enabling serial-to-parallel and parallel-to-serial conversions. Types include SISO, SIPO, PISO, and PIPO, with bidirectional and universal variants for added flexibility. Widely used in **data transmission, encryption, and digital signal processing**.
- **Finite State Machines (FSMs)**: Circuits that transition between states based on inputs and stored state, forming the backbone of **control logic** in CPUs, protocols, and embedded systems.

In summary, sequential circuits form the **memory and control layer** of digital systems. By combining storage elements with combinational logic, they enable digital systems to **process, remember, and react over time**—capabilities essential for CPUs, communication systems, and real-time controllers.

---

### 1.3 Register-Transfer Level (RTL)

Successful digital design begins not with coding, but with **planning**. Before writing any Verilog, a designer should first create hardware diagrams—such as block diagrams, datapath diagrams, finite state machine (FSM) charts, or control-signal tables. Aligning Verilog structure with these diagrams ensures that the implementation faithfully matches the intended functionality.

#### A Registered Incrementer Example

Consider the task of modeling a **registered incrementer** in Verilog or SystemVerilog. The design process typically includes:

- **Module interface definition** (`File.v`): specify ports such as an 8-bit input, 8-bit output, and clock input.
- **Internal signal declaration**: define `reg_out` (sequential storage) and `temp_wire` (combinational result).
- **Sequential logic**: update `reg_out` using `always @(posedge clk)` or `always_ff` with non-blocking assignments (`<=`).
- **Combinational logic**: compute `temp_wire = reg_out + 1` using `always @(*)` or an `assign` statement.
- **Output assignment**: drive the output port with `assign out = temp_wire`.
- **Verification**: write a **testbench** to check correctness through simulation.

This structured separation of sequential and combinational logic ensures clarity and consistency in the design.

#### Naming and Namespace Management

Unlike high-level languages such as Python, Verilog lacks built-in namespace management. This often leads to name conflicts for macros and modules. A practical convention is to use **path-based prefixes**:

- **Macros** → Uppercase with descriptive prefixes, e.g., `TUT4_VERILOG_REGINCR_...`
- **Modules** → Lowercase prefixes, e.g., `tut4_verilog_regincr_...`

Although verbose, this convention avoids collisions and improves maintainability in larger projects.

#### Coding Conventions

Readable and maintainable RTL depends on clear **interface design and naming conventions**:

- **Module names**: CamelCase, with each word capitalized.
- **Port names**: lowercase with underscores (snake_case).
- **Port declarations**: one per line, vertically aligned for clarity.
- **Identifiers**: descriptive and meaningful, avoiding ambiguous abbreviations.

This discipline ensures consistency across a project and makes RTL easier to understand and review.

#### Sequential vs. Combinational Logic

- **Registers (Sequential Logic)**: Modeled with `logic` variables and non-blocking assignments (`<=`) inside `always_ff` or `always @(posedge clk)`. Registers hold state and update on clock edges.

- **Combinational Logic**: Modeled using `always_comb` or continuous assignments (`assign`). Outputs depend only on current inputs. Simple logic, such as `assign out = reg_out + 1;`, can often be expressed directly without an `always` block.

**Signal roles in the registered incrementer**:

- `reg_out` → sequential storage (register state).
- `temp_wire` → combinational result (incremented value).

This separation helps prevent unintended latches and synthesis mismatches.

#### Testbench Design

Simulation validates RTL behavior before synthesis. A **testbench** typically includes:

- **Clock generation**:

  ```verilog
  initial clk = 0;
  always #5 clk = ~clk; // 10 time units per cycle
  ```

- **DUT instantiation**: use **named port binding**, with ports listed vertically for readability.
- **Initialization and stimulus**: an `initial` block applies test inputs, checks outputs, and reports mismatches using `$display`.
- **Waveform dumping**: `$dumpfile` and `$dumpvars` generate `.vcd` files for inspection in tools like GTKWave.

Importantly, `initial` blocks and delay-based clocks are **simulation-only** constructs. Real hardware must rely on **reset logic** for proper initialization.

#### Incremental Modifications

Changing RTL behavior should be straightforward. For example, to modify the incrementer from `+1` to `+2`:

```verilog
assign out = in_ + 8'd2;
```

The testbench is then updated with new expected results, ensuring correctness. This process reinforces a designer’s understanding of functional intent, while verifying changes through simulation.

#### PyMTL3 Integration

Writing testbenches directly in Verilog can be verbose. **PyMTL3**, a Python-based hardware modeling framework, provides a higher-level alternative. With **VerilogPlaceholder wrappers**, Verilog modules can be imported into PyMTL3 and tested using modern frameworks like **pytest**.

Advantages of PyMTL3 testbenches include:

- **Line tracing and debugging** for quick inspection.
- **VCD waveform generation** for GTKWave visualization.
- **Parameterized testing** with Python’s flexibility.

Wrappers map Verilog ports to PyMTL3 interfaces. Parameters such as bit-widths or module depth can be passed through metadata, enabling **design-space exploration** without rewriting RTL.

#### Hierarchical and Parameterized Design

Complex hardware is built by composing smaller RTL modules:

- **Structural composition**: instantiate multiple incrementer modules and connect them into multi-stage pipelines.
- **Parameterized modules**: use Verilog’s `parameter` and `generate` constructs to scale hardware (e.g., an N-stage incrementer).
- **PyMTL3 parameter passing**: dynamically configure Verilog modules during testing while reusing the same test framework.

This combination of parameterization and structural composition allows designs to be **scalable, reusable, and maintainable**, while static elaboration ensures hardware is determined at compile time—not runtime.

<details><summary>Code</summary>

```Verilog

// 1. Understanding Variable Types Fundamentally

    // Whether a logic variable models a register or wire depends on usage, not
  declaration:

    logic [7:0] reg_out;     // Used in sequential logic -> models register
    logic [7:0] temp_wire;   // Used in combinational logic -> models wire

    // reg_out models register: because it's updated in sequential blocks
    always_ff @(posedge clk) begin
        reg_out <= in_;
    end

    // temp_wire models wire: because it's updated in combinational blocks
    always_comb begin
        temp_wire = reg_out + 1;
    end



    /*
     * 2. Two methods for modeling combinational logic: continuous assignment vs
  always_comb blocks
     * Method 1: continuous assignment statement (recommended for simple logic)
     * - Direct mapping to hardware wires
     * - More intuitive, less error-prone
     */
    assign out = reg_out + 1;

    /*
     * Method 2: always_comb concurrent block (for complex logic)
     * - More flexible, supports complex control structures
     * - But easier to model unrealistic hardware
     */
    always_comb begin
        case (sel)
            2'b00: temp_wire = reg_out + 1;
            2'b01: temp_wire = reg_out + 2;
            2'b10: temp_wire = reg_out << 1;
            2'b11: temp_wire = reg_out;
        endcase
    end

    /*
     * 3. iverilog simulation: basic testbench structure
     * - include file inclusion
     * - clock generation
     * - device instantiation
     * - test sequence
     */

    module testbench;
        // Clock generation (non-synthesizable)
        logic clk = 1;
        always #5 clk = ~clk;           // 10 time unit period

        // Device instantiation - named port binding (recommended)
        RegIncr dut (
            .clk   (clk),
            .reset (reset),
            .in_   (in_),
            .out   (out)
        );

        // Test sequence
        initial begin
            $dumpfile("sim.vcd");       // Waveform file
            $dumpvars;                  // Dump all variables

            // Test cases
            reset = 1'b1;
            #11 reset = 1'b0;

            in_ = 8'h00;
            #10;
            if (out != 8'h01) begin
                $display("ERROR: expected %h, got %h", 8'h01, out);
                $finish;
            end

            $display("*** PASSED ***");
            $finish;
        end
    endmodule

    # PyMTL3 wrapper - connects Verilog modules with Python testing framework
    from pymtl3 import *
    from pymtl3.passes.backends.verilog import *

    class RegIncr(VerilogPlaceholder, Component):
        def construct(s):
            s.in_ = InPort(8)
            s.out = OutPort(8)

            # Optional: set port mapping
            # s.set_metadata(VerilogPlaceholderPass.port_map, {
            #     s.in_: 'in_',
            #     s.out: 'out',
            # })

    /*
     * 4. Structural composition - module reuse
     * - include submodules
     * - instantiation and connection
     * - named port binding
     */

    module RegIncr2stage(
        input logic clk, reset,
        input logic [7:0] in_,
        output logic [7:0] out
    );
        logic [7:0] stage1_out;         // Intermediate signal

        // First stage
        RegIncr stage1 (
            .clk   (clk),
            .reset (reset),
            .in_   (in_),
            .out   (stage1_out)
        );

        // Second stage
        RegIncr stage2 (
            .clk   (clk),
            .reset (reset),
            .in_   (stage1_out),
            .out   (out)
        );
    endmodule

    /*
     * Parameterized module - design reuse
     * - parameter declaration
     * - generate statements
     * - compile-time static elaboration
     */
    module RegIncrNstage #(
        parameter p_nstages = 2
    )(
        input logic clk, reset,
        input logic [7:0] in_,
        output logic [7:0] out
    );
        // Signal array - stores inter-stage connections
        logic [7:0] stage_out [p_nstages+1];

        assign stage_out[0] = in_;      // Input connection

        // Generate block - static elaboration
        genvar i;
        generate
            for (i = 0; i < p_nstages; i++) begin: gen_stages
                RegIncr stage_inst (
                    .clk   (clk),
                    .reset (reset),
                    .in_   (stage_out[i]),
                    .out   (stage_out[i+1])
                );
            end
        endgenerate

        assign out = stage_out[p_nstages];  // Output connection
    endmodule
```

</details>

---

## 2. FPGA

Whether you are working with an FPGA or an ASIC, the design process begins the same way: writing RTL (Register Transfer Level) code in an HDL such as Verilog, VHDL, or SystemVerilog. At the code level, there is no semantic difference—HDL describes combinational logic, sequential logic, and storage.

The key distinction lies not in the language, but in the **target architecture**. FPGA and ASIC flows diverge because the RTL is mapped to fundamentally different physical resources:

- **FPGA** → mapped to programmable resources (LUT, FF, BRAM, DSP) → constrained by **architectural limits**
- **ASIC** → mapped to standard cell libraries and physical circuits → constrained by **process and physical implementation**

### FPGA Mapping

FPGA synthesis tools map HDL logic to **look-up tables (LUTs)**, **flip-flops (FFs)**, **block RAM (BRAMs)**, **DSP slices**, and **I/O cells**. For example,

```verilog
assign y = a & b & c & d;
```

might be implemented as a single 4-input LUT or decomposed into multiple cascaded LUTs.

When writing RTL for FPGA, the designer must be aware of the **fixed architectural resources**:

- LUT width and count are limited
- Each LUT comes with a fixed number of flip-flops
- BRAM/DSP blocks are finite and have fixed widths
- I/O pins are limited by the package
- Global clock trees are few in number
- Routing congestion can prevent timing closure

FPGA-friendly RTL often requires following **vendor-specific templates**. For example:

- Writing `reg [31:0] mem [0:255];` allows tools to infer BRAM.
- Using `assign y = a * b;` helps the compiler map the multiplier into a DSP slice.

In short, **FPGA coding must adapt to architectural constraints.**

### ASIC Mapping

ASIC synthesis tools, by contrast, map RTL logic to **standard cell libraries**: gates (AND, OR, INV), flip-flops, and registers defined for a given process node.

The same logic example above is synthesized into an optimal combination of gates, with **no LUT restrictions**. RTL for ASICs is therefore **closer to ideal Boolean logic**—but the constraints shift toward physical design. Designers must account for:

- Clock tree synthesis (CTS)
- Power optimization
- Timing closure across corners
- Chip area and routing congestion
- Power integrity (IR drop), signal integrity, and EMI
- Fabrication costs and yield considerations

In ASICs, memory is not inferred from generic RTL. Instead, **SRAM macros** are generated using a memory compiler. In some cases, hand-crafted gate-level optimizations or explicit cell instantiations are needed.

Thus, ASIC design enjoys more **freedom in logic expression**, but requires greater attention to **physical implementation and process technology** (e.g., TSMC 5nm vs. 28nm libraries).

### Structural Comparison

- **FPGA Architecture**

  - Basic unit: LUT + FF
  - Memory: on-chip BRAM or distributed RAM (using LUTs)
  - Computation: limited DSP slices (multipliers, MACs)
  - Connectivity: programmable routing matrix
  - Characteristic: **fixed architecture, circuits are “assembled” through configuration bits**

- **ASIC Architecture**

  - Basic unit: gates and flip-flops (from a standard cell library)
  - Memory: SRAM/ROM macros, fully customizable size/organization
  - Computation: fully optimized arithmetic units or custom blocks
  - Connectivity: metal interconnect layers (fixed once taped out)
  - Characteristic: **true physical circuits with no LUT abstraction**

In essence, FPGA RTL must “fit the fabric,” whereas ASIC RTL must “fit the physics.”

<details><summary>Code</summary>

```VHDL

---------------------------------------------------
-- Non-linear Lookup Table Implementation in VHDL--
---------------------------------------------------
entity non_linear_lookup is
port (  LUTIN: in std_logic_vector(7 downto 0);
   LUTOUT: out std_logic_vector(7 downto 0)
 );
end non_linear_lookup;

architecture Behavioral of non_linear_lookup is
signal MSN_in,LSN_in,MSN_out,LSN_out: std_logic_vector(3 downto 0);
begin
MSN_in <= LUTIN(7 downto 4);
LSN_in <= LUTIN(3 downto 0);
SBOX_1: process(MSN_in) begin
case(MSN_in) is
 when "0000" => MSN_out <= "0001";
 when "0001" => MSN_out <= "1011";
 when "0010" => MSN_out <= "1001";
 when "0011" => MSN_out <= "1100";
 when "0100" => MSN_out <= "1101";
 when "0101" => MSN_out <= "0110";
 when "0110" => MSN_out <= "1111";
 when "0111" => MSN_out <= "0011";
 when "1000" => MSN_out <= "1110";
 when "1001" => MSN_out <= "1000";
 when "1010" => MSN_out <= "0111";
 when "1011" => MSN_out <= "0100";
 when "1100" => MSN_out <= "1010";
 when "1101" => MSN_out <= "0010";
 when "1110" => MSN_out <= "0101";
 when "1111" => MSN_out <= "0000";
 when others => MSN_out <= "0000";
end case;
end process;
SBOX_2: process(LSN_in) begin
case(LSN_in) is
 when "0000" => LSN_out <= "1111";
 when "0001" => LSN_out <= "0000";
 when "0010" => LSN_out <= "1101";
 when "0011" => LSN_out <= "0111";
 when "0100" => LSN_out <= "1011";
 when "0101" => LSN_out <= "1110";
 when "0110" => LSN_out <= "0101";
 when "0111" => LSN_out <= "1010";
 when "1000" => LSN_out <= "1001";
 when "1001" => LSN_out <= "0010";
 when "1010" => LSN_out <= "1100";
 when "1011" => LSN_out <= "0001";
 when "1100" => LSN_out <= "0011";
 when "1101" => LSN_out <= "0100";
 when "1110" => LSN_out <= "1000";
 when "1111" => LSN_out <= "0110";
 when others => LSN_out <= "0000";
end case;
end process;
LUTOUT <= MSN_out & LSN_out;
end Behavioral;
```

</details>

---

## 3. GPU Programming with CUDA C/C++

CUDA (Compute Unified Device Architecture) is NVIDIA’s parallel computing platform and programming model for GPUs. Built as an **extension to C/C++**, it provides developers with a direct interface to harness the massive parallelism of modern GPUs. In practice, programmers write CUDA kernels—functions annotated with `__global__`—and dispatch them for execution on the GPU, while the rest of the program runs as normal C/C++ code on the CPU.

### CUDA Execution Model: Threads, Blocks, and Grids

CUDA organizes computation hierarchically:

- **Thread** – the smallest execution unit, each running one instance of the kernel code.
- **Block** – a group of threads, typically arranged in 1D, 2D, or 3D layouts. Threads within a block share fast on-chip memory.
- **Grid** – a collection of blocks, defining the overall execution configuration across the GPU.

The kernel launch syntax `kernel<<<gridDim, blockDim>>>(...)` specifies how many blocks (`gridDim`) and how many threads per block (`blockDim`) will run.

### SIMT Execution and Memory Hierarchy

CUDA follows the **SIMT (Single Instruction, Multiple Threads)** model: threads execute in groups of 32, called **warps**, all running the same instruction in lockstep. This model enables extreme parallelism but introduces challenges such as branch divergence.

The GPU memory hierarchy is a key factor in performance:

- **Registers** – private to each thread, fastest storage.
- **Shared Memory** – accessible by all threads in a block, located on-chip with very low latency.
- **Global Memory** – accessible by all threads, large but high latency.
- **Constant and Texture Memory** – optimized for specific access patterns.
- **Local Memory** – private to a thread but physically stored in global memory, often used for register spilling.
- **L1/L2 Cache** – hardware-managed cache layers.

Choosing the right memory for each data structure is critical to minimizing latency and maximizing throughput.

### Performance Optimization Strategies

The main performance bottlenecks often come from memory access. Key techniques include:

- **Coalesced Memory Access** – aligning thread accesses so that consecutive threads access consecutive memory locations, allowing memory requests to be merged into fewer transactions.
- **Minimizing Warp Divergence** – since all threads in a warp must follow the same instruction path, divergent branches cause serial execution and reduce throughput.
- **Streams and Asynchronous Execution** – CUDA streams allow overlapping of data transfers and computation. Asynchronous kernel launches and memory copies reduce CPU-GPU synchronization overhead.
- **Optimizing CPU–GPU Data Transfer** – data movement across the PCIe bus is often a bottleneck. Techniques include using **pinned memory** for faster transfers, reusing GPU-resident data as much as possible, and leveraging **Unified Memory** to simplify memory management.
- **Low-Level PTX Optimization** – when higher-level optimization isn’t enough, developers can inspect the generated **PTX (Parallel Thread Execution)** assembly. Analyzing register usage, instruction scheduling, and memory access patterns in PTX helps fine-tune kernels for maximum efficiency, especially in compute-intensive domains like cryptography or numerical methods.

### Practical Considerations in ZK Accelerator Development

In real-world zero-knowledge (ZK) accelerator projects, developers typically write three categories of code:

- **API Calls**: such as memory management (`cudaMalloc(&ptr, size)`), which rely on CUDA’s runtime library.
- **CUDA Kernels**: core algorithms annotated with `__global__`, implementing parallel ZK primitives.
- **Application Logic**: the higher-level implementation of ZK protocols, orchestrating GPU computation.

Other components—such as the CUDA SDK, GPU drivers, or Linux system calls—are already provided and do not need to be reimplemented by developers.

<details><summary>Code</summary>

```Cuda

// 1. CUDA Runtime Layer - Direct API Calls
  // Calling NVIDIA provided functions
  #include <cuda_runtime.h> // NVIDIA provided header file

  void my_gpu_function() {
      float *d_data, *h_data;
      dim3 grid, block;
      size_t size = 1024 * sizeof(float);

      h_data = (float*)malloc(size);
      grid = dim3((1024 + 255) / 256);
      block = dim3(256);

      // Call NVIDIA's cudaMalloc
      cudaError_t err = cudaMalloc(&d_data, size);

      // Call NVIDIA's cudaMemcpy
      cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);

      // Launch our own kernel
      my_kernel<<<grid, block>>>(d_data);

      // Call NVIDIA's synchronization function
      cudaDeviceSynchronize();

      // Call NVIDIA's free function
      cudaFree(d_data);

  }

  // 2. System Call Layer - Indirect calls through library functions

  #include <unistd.h> // System provided header files
  #include <fcntl.h>

  void file_operation() {
  // Call glibc provided functions, glibc will call system calls
      struct nvidia_info info;
      int fd = open("/dev/nvidia0", O_RDWR); // glibc implementation, eventually calls
  sys_open

      // Write our own calling code, but ioctl function is provided by glibc
      ioctl(fd, NVIDIA_IOCTL_CARD_INFO, &info);  // glibc -> sys_ioctl

      close(fd);  // glibc implementation, eventually calls sys_close

  }

  // 3. Dynamic Library Linking Calls
  // Compile-time linking
  // In Makefile: gcc -lcuda -lcudart my_program.c

  // At runtime, system automatically loads these libraries:
  /*
  libcuda.so.1 <- NVIDIA written user-space driver
  libcudart.so.11 <- NVIDIA written Runtime library
   */

  // Our own code only needs to call
  cudaMalloc(&ptr, size); // This will automatically call functions in libcudart.so

  // 4. GPU_FFT.cu - libSTARK code
  #include "GPU_FFT.cuh"
  #include <cuda_runtime.h> // NVIDIA provided
  #include <cuda.h> // NVIDIA provided

  void GPU_FFT::fft_gpu(const FFT* fft, Polynomial* P) {
      Chunk * d_chunk_P;
      len_t p_len = P->getLength();

      // Call NVIDIA's API
      cudaMalloc(&d_chunk_P, sizeof(Chunk) * p_len);
      cudaMemcpy(d_chunk_P, P->getCoeffs(), sizeof(Chunk) * p_len,
                  cudaMemcpyHostToDevice);

      // Launch our own kernel
      fft_kernel<<<grid, block>>>(d_chunk_P, p_len);

      // Call NVIDIA's API again
      cudaMemcpy(P->getCoeffs(), d_chunk_P, sizeof(Chunk) * p_len,
                  cudaMemcpyDeviceToHost);
      cudaFree(d_chunk_P);

  }

  // CUDA kernel we need to write ourselves
  __global__ void fft_kernel(Chunk* data, len_t len) {
      // Your FFT algorithm implementation
      int idx = blockIdx.x * blockDim.x + threadIdx.x;
      if (idx < len) {
      // Specific FFT computation logic
      }
  }

  // 5. Compilation and Linking Process
  // Parts that need configuration:
  // - Compile and link options (-lcudart -lcuda)
  // - Header file inclusion (#include <cuda_runtime.h>)
  // - Library file paths (-L/usr/local/cuda/lib64)

  # Need to specify library and header file paths during compilation

  nvcc -I/usr/local/cuda/include \ # CUDA header files
  -L/usr/local/cuda/lib64 \ # CUDA library file paths
  -lcudart -lcuda \ # Link CUDA libraries
  GPU_FFT.cu -o gpu_fft

```

</details>

---

[HDLBits](https://hdlbits.01xz.net/wiki/Main_Page0)
[EDA Playground](https://edaplayground.com/)
[DigitalJS](https://digitaljs.tilk.eu/)
[ChipVerify](https://www.chipverify.com/)
[ZPrize](https://www.zprize.io/)
[Icicle-gpu](https://github.com/ingonyama-zk/icicle)
