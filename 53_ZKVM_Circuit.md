# ETAAcademy-ZKMeme: 53. ZKVM Circuit

<table>
  <tr>
    <th>title</th>
    <th>tags</th>
  </tr>
  <tr>
    <td>53. ZKVM Circuit</td>
    <td>
      <table>
        <tr>
          <th>zk-meme</th>
          <th>basic</th>
          <th>quick_read</th>
          <td>ZKVM Circuit</td>
        </tr>
      </table>
    </td>
  </tr>
</table>

[Github](https://github.com/ETAAcademy)｜[Twitter](https://twitter.com/ETAAcademy)｜[ETA-ZK-Meme](https://github.com/ETAAcademy/ETAAcademy-ZK-Meme)

Authors: [Evta](https://twitter.com/pwhattie), looking forward to your joining

# zkVM Circuit Design and Key Technologies

zkVM (Zero-Knowledge Virtual Machine) is a specialized virtual machine that aims to transform the execution of an Instruction Set Architecture (ISA) into arithmetic circuits, making the process verifiable using Zero-Knowledge Proofs (ZKPs), such as Groth16, Plonk, or STARK. To illustrate the power of this approach, two examples—zkWasm and SP1—highlight the versatility and efficiency of zkVM circuits in different contexts.

zkWasm applies arithmetic circuits and ZK-SNARKs to WebAssembly (WASM), utilizing advanced techniques like sharding, batching, and recursive proofs to ensure secure, privacy-preserving, and scalable verification of WASM execution. Similarly, SP1 focuses on RISC-V program verification, leveraging a modular multi-table architecture and advanced protocols to enhance scalability, efficiency, and security in decentralized applications.

---

The core task of a zkVM is to convert the execution process of a program (represented by ISA instructions) into a set of verifiable arithmetic constraints, ensuring that every operation performed can be proven to be correct without exposing the underlying data. Unlike traditional virtual machines, such as EVM (Ethereum Virtual Machine) or WASM (WebAssembly), the zkVM has specific requirements for its instruction set:

- **Proofability**: Every instruction must be represented by a mathematical constraint in the arithmetic circuit.
- **Low Overhead**: The circuit’s complexity should be minimized to reduce proof generation time.
- **Compatibility**: It must support existing programming languages (e.g., Solidity, Rust) and their compilers.
- **Dynamic Control Flow**: The presence of loops and conditional branches may cause circuit size to explode, which must be optimized through static analysis or dynamic path constraints.
- **State Consistency**: Memory and register reads/writes must maintain consistency across the circuit.
- **Cross-Instruction Constraints**: Dependencies between instructions need to be modeled as global constraints to prevent vulnerabilities.
- **LogUp Protocol**: This protocol uses "logarithmic derivatives" to ensure data consistency across multiple tables, reducing computational overhead.
- **Sharding**: Large programs can be split into smaller circuits, each independently proving its correctness before being recursively merged.

The architecture of zkVM is often modular, divided into multiple tables where each table serves a different purpose. Communication between these tables ensures global consistency in the computation.

| **Table Type**         | **Function**                                                                                                                                                          | **Example Constraints**                                                 |
| ---------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------- |
| **CPU Table**          | Records basic information of instruction execution (PC, opcode, register values).                                                                                     | `pc_next = pc + 1` (Program Counter Increment)                          |
| **ALU Table**          | Handles basic arithmetic instructions such as addition, multiplication, boolean and bitwise operations, directly mapped to finite field constraints.                  | For the RISC-V `ADD` instruction: `a + b - c = 0` (Addition constraint) |
| **Memory Table**       | Manages memory operations like `LOAD` and `STORE`, often optimized using Merkle trees or Sparse Merkle Trees.                                                         | `memory[addr] = old_value → new_value` (Memory update constraint)       |
| **Control Flow Table** | Handles control flow instructions like `JUMP` and `CALL`, ensuring the program counter is correctly updated.                                                          | `if (condition) { pc_next = target } else { pc_next = pc + 1 }`         |
| **Precompiled Table**  | Optimizes complex operations like elliptic curve operations (ECADD, ECMUL) and hash functions (SHA256, Poseidon) by precompiling them into efficient circuit modules. | `SHA256(input) = output` (Precompiled hash function constraint)         |

### **Key Techniques in zkVM Circuit Design**

The design of the zkVM circuit is the backbone of Zero-Knowledge Proof generation. The circuit encodes the execution process of a program as a set of polynomial constraints, which directly impact the efficiency of proof generation and verification.

### 1) **Circuit Layering Architecture**

- **Logic Layer**: Simulates the traditional virtual machine's execution, including instruction decoding, register operations, and memory accesses.
- **Constraint Layer**: Converts the logic operations into arithmetic circuit constraints (e.g., R1CS or Plonkish constraints).
- **Proof Layer**: Integrates the ZKP protocol (such as Groth16 or STARK) to generate verifiable proofs.

### 2) **Core Circuit Modules**

- **Execution Trace Circuit**: Tracks every state change during program execution (e.g., register, memory, program counter), encoding these states as polynomials. For control flow constraints, it handles operations like conditional jumps or function calls, where the program counter (PC) must change dynamically.

  - **Branching**: The constraint equation would be: `pc_next = condition ? target : pc + 1`, verifying the `condition` logic (e.g., `BEQ` comparison).
  - **Function Call/Return**: Stack operations are constrained by pushing return addresses when calling functions and popping them when returning, with the memory table tracking changes to the stack pointer.

- **Memory Access Circuit**: Memory access validation is one of the biggest challenges in zkVM design. Merkle trees or lookup tables are often used to verify correct memory reads and writes, ensuring consistency across the system.

  - **Merkle Patricia Trie (MPT)**: Memory states are represented by Merkle tree hashes. Every memory read and write updates the root hash. The constraint ensures the validity of the Merkle path before and after the operation.
  - **Memory Table Design**: Each entry records `(address, old_value, new_value, operation_type)`. The constraint ensures that if the operation is a write, `new_value = computed_value`; if it’s a read, `old_value = current_memory_value`.

- **Precompiled Circuit**: For high-frequency operations like hashing and signature verification, zkVM uses precompiled circuits to reduce constraint overhead, ensuring that these complex operations are handled efficiently in the underlying circuit.

### 3) **Optimization Strategies**

| **Technology**             | **Principle**                                                                                                                                                          | **Application Example**                                  |
| -------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------- |
| **Custom Gates**           | Encapsulate high-frequency operations (e.g., hashing, loops) into specialized gates to reduce constraint count.                                                        | zkSync’s Keccak hash precompiled circuit.                |
| **Lookup Tables**          | Precompute complex functions (e.g., bitwise operations) into input-output pairs, replacing real-time computation with table lookup.                                    | Range-check optimizations in Halo2.                      |
| **Recursive Proofs**       | Break down large-scale computations into smaller circuits, each producing a proof, which are then recursively combined. This reduces the complexity of a single proof. | RISC Zero’s shard execution and proof combination.       |
| **Constraint Compression** | Merge redundant constraints, for example, simplifying memory address verification with range checks.                                                                   | Range check optimizations in zkVM.                       |
| **Circuit Parallelism**    | Parallelize independent operations, such as multiple memory accesses, to generate proofs in parallel.                                                                  | Parallel proof generation in zkVM.                       |
| **Hardware Acceleration**  | Utilize GPUs or FPGAs to accelerate polynomial calculations like FFT.                                                                                                  | Hardware acceleration in zkVM for polynomial operations. |

---

## 1. zkWasm: Enabling Privacy and Verifiable Computation for WebAssembly

zkWasm is an implementation of zkVM technology specifically designed for the **WebAssembly (Wasm)** instruction set. Its primary goal is to provide privacy and verifiable computation for high-performance blockchain applications, such as games and high-concurrency decentralized applications (DApps). WebAssembly is a cross-platform binary instruction format originally developed to improve webpage performance, but it has since become a mainstream technology for running serverless functions in cloud services like AWS Lambda. However, its native runtime does not inherently support **privacy-preserving computation** (e.g., ensuring input data remains confidential) or **trusted verification** (e.g., proving the correctness of computational results).

**ZAWA**, the ZK-SNARK-based Wasm virtual machine, enables **zero-knowledge proof generation** without modifying existing Wasm code. By deeply integrating Wasm runtime with ZK-SNARK, ZAWA’s core innovation lies in systematically encoding the complex virtual machine semantics into arithmetic circuit constraints. This enables a **"privacy + verifiable"** solution for serverless computing, allowing computations to be performed with guarantees of privacy and correctness.

### **Technical Features**

#### 1) **Wasm Instruction Set**

Wasm is a stack-based virtual machine with complex state operations, including memory management and control flow. The challenge is to map these operations into the ZK-SNARK constraint system, representing each Wasm instruction as a circuit logic and ensuring the execution of each instruction is verifiable through polynomial constraints.

- **Compact Binary Format**: Wasm’s binary format is designed for fast execution and supports multiple language compilations (e.g., Rust, C++).
- **Wide Application**: It is widely used in blockchain ecosystems (e.g., NEAR, Polkadot) and high-performance web scenarios.

#### 2) **Zero-Knowledge Proof (ZKP) Integration**

ZAWA integrates ZKPs into the Wasm virtual machine by abstracting Wasm instructions into polynomial constraints. Key features include:

- **Instruction Classification**: Wasm instructions, such as arithmetic operations, memory access, and function calls, are abstracted into circuit logic, ensuring that the execution of each instruction is expressed via polynomial constraints.
- **Global State Modeling**: Circuit variables simulate the Wasm runtime state, such as the stack, memory, global variables, ensuring that every state transition adheres to the expected behavior.
- **Performance Optimization**:
  - **Sharding and Batching**: Long execution traces (e.g., complex calculations) are split into multiple partitions. Proofs for each partition are generated in parallel and later combined (batching), reducing computational overhead.
  - **External Instruction Extensions**: ZAWA supports precompiled high-efficiency circuit modules (e.g., cryptographic algorithms) through the Foreign Function Interface (FFI), which helps reduce the circuit size for general instructions.

The Wasm virtual machine can be abstracted as a deterministic state machine, where the state is composed of:

- **Instruction Address (`iaddr`)**: Points to the current executing instruction in the code segment.
- **Call Stack Frame (`F`)**: Tracks function call depth and manages function calls and returns.
- **Memory (`M`)**: Linear memory space for dynamic data storage.
- **Global Variables (`G`)**: Shared state across functions.
- **Operand Stack (`SP`)**: Stores intermediate values in stack-based computations.
- **I/O State (`IO`)**: Input-output buffers, such as stdin and stdout.

The execution process is described by a state transition function ($t_i: S → S$) that transitions the current state `s` to a new state `s'`. Each Wasm instruction corresponds to a state transition, forming an execution trace $[t_0, t_1, t_2, \dots]$. Given the Wasm virtual machine's input $(I(C, H), E, IO)$ and initial state $s_0 (s_0.iaddr = E)$, the valid execution trace must meet the requirements of correct initial instruction semantics, consistent state transitions, and termination conditions (e.g., call stack depth reaches 0). The correctness of the final output depends on the existence of a valid execution trace that adheres to Wasm semantics, forming the theoretical foundation for later mapping Wasm semantics into arithmetic circuits and constructing the zero-knowledge proof system.

### **Core Goal of ZAWA**

ZAWA’s main objective is to compile the execution process of the Wasm virtual machine into arithmetic circuits and generate ZK-SNARK proofs. The process includes several key steps:

#### 1) **Circuit Design**

Wasm instructions are mapped to polynomial constraints to construct the arithmetic circuit. The program is compiled into an arithmetic circuit represented by constraints.

- Convert the program $P$ into an arithmetic circuit $C$ made up of polynomial constraints, ensuring that for any input $params$, there exists a unique set of intermediate variables $w$ and output $r$, such that $C(params, w, r) = 0$.
- The task of proving $P(params) = r$ is transformed into proving that the vector $v = (params, w, r)$ satisfies the circuit constraint $C(v) = 0$.

#### 2) **Proof Generation**

Run the Wasm virtual machine, record the execution trace, and populate the circuit matrix to generate a ZK-SNARK proof that satisfies the constraints.

- **Polynomial Constraints and Polynomial Commitment**: The circuit constraint $C$ is turned into a polynomial evaluation problem. The goal is to find a polynomial $p$ and a set of point-value pairs $(x_i, v_i)$ such that $p(x_i) = v_i$ implies $C$.
  - **Example**: A linear constraint $\sum c_{ij} x_j = 0$ can be transformed into a polynomial equation $\sum c_i(X) p(X) = 0$, proving that the polynomial evaluates to 0 at specific points.
- **Polynomial Commitment Scheme (PCS)**: A scheme like KZG is used to efficiently verify the polynomial's value at specific points.

#### 3) **Verification Process**

The verifier checks the validity of the proof, confirming the correctness of the computation without needing to reproduce the computation itself.

---

### 1.1 Arithmetic Circuits

Arithmetic circuits are the core component of zero-knowledge proofs (ZKPs) and serve as the foundational mathematical structure for ZK-SNARKs. Their purpose is to encode the logic of a program into polynomial constraints, enabling the formal verification of the correctness of a computation process while ensuring **no privacy data is leaked**. In the context of the ZK-SNARK WASM simulator ZAWA, arithmetic circuits are used for the following purposes:

- **Encoding WASM Instructions**: Each WASM instruction (such as `i32.add`, `local.get`) is mapped to polynomial constraints.
- **Memory Access Verification**: Lookup constraints are used to ensure memory reads and writes align with historical records (e.g., `MTable`).
- **Range Checking**: Verifying that variable values remain within a legal range (e.g., for 32-bit integers).

The Halo2 arithmetic circuit system encodes program logic into verifiable mathematical forms using **polynomial equations** and **lookup constraints**. Among various arithmetic circuit systems (like Plonk, Groth16), **Halo2** is chosen as the foundational framework for ZAWA, particularly for the ZK-SNARK WASM simulator. Halo2 is preferred because it supports custom constraint types and is well-suited for the complex structure of the WASM virtual machine. Halo2's **polynomial lookup** functionality enables efficient implementation of table lookups (such as memory tables like `MTable`) and range checks. Furthermore, its support for **recursive proofs** allows for scalable zero-knowledge proofs without a trusted setup.

Halo2 circuits consist of two main components:

- **Matrix (Grid)**: An $n$-column table, where each row corresponds to one step in the program execution, and the columns correspond to different types of variables (such as operation codes, memory addresses, operation results, etc.).
- Each cell in the circuit matrix $G$ is denoted as $G_{l, c, r}$, where $l$ is the current row index, $c$ is the column index, and $r$ is the relative row offset (e.g., for adjacent rows).

Each constraint $C_i$ in the constraint system $\mathcal{C}$ is one of the following two forms:

- **Polynomial Equation Constraint**: $P(G_{l, c_0, r_0}, G_{l, c_1, r_1}, \dots, G_{l, c_k, r_k}) = 0$, where $P$ is a fixed polynomial ensuring mathematical relationships between variables (such as addition or multiplication).
- **Polynomial Lookup Constraint**: $(G_{l, c_0, r_0}, G_{l, c_1, r_1}, \dots, G_{l, c_k, r_k}) \in \mathcal{T}$, where the combination of variables must exist in a predefined table $\mathcal{T}$ (e.g., memory tables or range tables).

### **Example Analysis: Circuit Design for the Sum Function**

Let’s consider an example to illustrate how Halo2 circuits encode program logic. Here, we use a simple summation function.

**Circuit Matrix Design** (Table 1):

| Column `s` | Column `acc` | Column `operand` |
| ---------- | ------------ | ---------------- |
| 1          | 0            | $v_0$            |
| 1          | $v_0$        | $v_1$            |
| 1          | $v_0 + v_1$  | $v_2$            |
| 0          | $\sum v_i$   | nil              |

**Constraint System Design**:

The constraints for the circuit are given as:

$\mathcal{C}(cur) =
\begin{cases}
s_{cur} \times (acc_{cur} + operand_{cur} - acc_{cur+1}) = 0 \\
s_{cur} \times (1 - s_{cur}) = 0
\end{cases}$

- **Addition Constraint**: When `s = 1`, it enforces $acc_{cur+1} = acc_{cur} + operand_{cur}$, implementing the summation logic. When `s = 0`, it ensures that the final value of `acc` remains unchanged, signaling the end of the loop.
- **Binary Constraint**: The value of the `s` column must be either 0 or 1, ensuring the correctness of the loop control.

**Application of Arithmetic Circuits in ZAWA**

In the ZK-SNARK WASM simulator ZAWA, arithmetic circuits are used for several key functions:

- **Encoding WASM Instructions**: Each WASM instruction (e.g., `i32.add`, `local.get`) is mapped into polynomial constraints.
- **Memory Access Verification**: Lookup constraints ensure that memory reads and writes conform to historical records (such as `MTable`).
- **Range Checking**: Verifying that variables remain within valid ranges (e.g., ensuring 32-bit integer constraints are upheld).

ZAWA utilizes arithmetic circuits and polynomial lookup tables to ensure the correctness of the WASM virtual machine execution. Key modules include data type representation, memory access validation, and mathematical semantics encoding. For example, in the Halo2 constraint system, WASM types such as `i32`/`i64` are verified through range checks and large number decomposition to ensure values stay within the elliptic curve scalar field and do not overflow. This is accomplished by defining tables such as `TN` and using lookup constraints to validate if variables lie within the defined range.

Additionally, ZAWA uses polynomial lookup tables to represent key-value mappings, simplifying the processing of dynamic data structures. Code and heap tables are constructed to verify the correctness of WASM virtual machine instructions and memory initialization, ensuring that addresses match with operation codes or values. Furthermore, ZAWA translates the mathematical semantics of WASM instructions (e.g., unsigned integer division) into arithmetic circuit constraints. The constraint system ensures that instructions are accurately mapped to mathematical operations, and tools like the Z3 theorem prover are used to verify their correctness.

Finally, the validity of memory accesses is guaranteed through sorting and constraints. Transactions involving memory operations are recorded along with their IDs, addresses, and operation types. A sorting table is constructed to ensure that read operations return the correct value based on the last write and to prevent any tampering with intermediate states.

---

### 1.2 ZAWA Architecture Circuits

ZAWA divides the WASM execution verification process into four distinct phases, utilizing multiple circuits working together to achieve zero-knowledge proofs:

- **Image Setup**: The code section of the WASM image is encoded into the lookup table TC, and the data section is encoded into the TH table. The TC table is used to validate the legality of the instructions in the execution trace, while the TH table ensures that memory initialization aligns with the image data.
- **Execution Trace Generation**: The WASM interpreter generates a sequence of instruction executions (transition function trace). The interpreter itself does not need to be trusted, as subsequent circuits will rigorously validate the correctness of the trace.
- **Synthesis Circuits**: The execution trace is populated into the main execution circuit TE, along with other tables such as the **TF table** (which manages function call frames), **TM table** (which logs memory accesses), and **TG/TSP table** (which tracks global variables and stack accesses). Constraints ensure that all operations conform to WASM semantics.
- **Proof Generation**: Using the Halo2 ZK-SNARK system, a proof is generated to verify the legitimacy of the execution trace and its output.

#### Core Circuit Breakdown

##### 1) Setup Circuits

- **Code Table TC**

  - **Structure**: Each instruction is uniquely identified by a tuple **`iaddr = (moid, mmid, fid, iid)`**, where `(moid, mmid, fid, iid) → opcode`. Here, **moid** is the module ID (distinguishing different WASM modules), **mmid** is the memory block instance ID (distinguishing memory blocks within a module), **fid** is the function ID (distinguishing functions within the module), and **iid** is the instruction offset (the sequence number of the instruction within the function). The purpose is to ensure that each instruction in the execution table corresponds to a valid instruction in the WASM code section.
  - **Constraint Rule**: Every instruction in the execution circuit TE must exist in the TC table:  
    $\forall e \in T_E, \quad (e.\text{iaddr}, e.\text{opcode}) \in T_C$
  - **Example**:

    | moid | mmid | fid  | iid  | opcode |
    | ---- | ---- | ---- | ---- | ------ |
    | 0x00 | 0x01 | 0x01 | 0x00 | add    |

- **Initial Data Table TH**

  - **Structure**: Records the initial state of memory and global variables, and constrains the legality of runtime memory accesses. The tuple `(ltype, mmid, offset) → (value, isMutable)` is used, where **offset** refers to the variable's position within the memory block.
  - **Constraint Rule**: The initialization entries in the memory log table TM must match the image data:  
    $\forall e \in T_M, \quad e.\text{accessType} = \text{Init} \implies (e.\text{iaddr}, e.\text{value}) \in T_H$
  - **Example**:  
    **ltype** distinguishes between heap and global variables, **value** is the initial value (64-bit unsigned integer), and **isMutable** marks whether the value is mutable (e.g., for constants or variables).

    | ltype | mmid  | offset | value | isMutable |
    | ----- | ----- | ------ | ----- | --------- |
    | Heap  | mmid0 | 1      | 0x01  | true      |

##### 2) Execution Trace Circuits

The execution trace circuit is the core component of ZAWA, responsible for encoding the execution trace generated by the WASM interpreter into circuit constraints, ensuring that each operation adheres to the WASM semantics.

- **Execution Table TE Structure**: The execution trace consists of a series of instruction blocks, each corresponding to the execution of a single WASM instruction. These blocks record micro-operations, with key columns including:

  - `start`: Marks the beginning of a new instruction
  - `opcode`: The opcode and micro-operations (e.g., load, write)
  - `address`: The address associated with the TC table's instruction address
  - `sp`: The stack pointer change
  - **Example**:

    | start | opcode | address (TC Index) | sp  | ... |
    | ----- | ------ | ------------------ | --- | --- |
    | true  | add    | (0x00, 0x01, ...)  | 100 | ... |
    | false | load   | 0x1234             | 96  | ... |

- **Key Constraints**: For each instruction, the pair `(iaddr, opcode)` must exist in the TC table. When operands are fetched from the stack, there must be a corresponding record in the memory log table TM. Additionally, all numeric fields (such as u64) must satisfy range constraints ($< 2^{64}$).

##### 3) Frame Circuits

The frame circuit is a critical component of the ZAWA architecture, used to manage function call stack frames and ensure the correctness of function calls and returns. Its primary function is to record the hierarchical relationship of call frames and verify the correct return address when a `return` instruction is encountered. The frame circuit uses a lookup table, where each row records metadata for a call frame:

- **TF Table Structure**: `(prev_frame, current_frame, iaddr)`
  - `current_frame`: The instruction ID (tid) of the current call frame
  - `prev_frame`: The tid of the previous call frame
  - `iaddr`: The address of the current call instruction
- **Constraint**: When executing a `return`, the return address is verified by looking it up in the TF table:  
  $\text{plookup}(T_F, (\text{prevFrame}, \text{currentFrame}, s_{i+1}.(\text{iaddr} - 1))) = 0$

##### 4) Access Log Circuits

The access log circuit tracks the memory, stack, and global variable accesses. This ensures that memory accesses and updates occur in a valid sequence, maintaining the integrity of the execution process.

- **TM Table Structure**: Tracks memory, stack, and global variable accesses, sorted by `(address, (tid, tmid)` to ensure access order validity. Fields include access type (Init/Read/Write), address, value, etc.

  - **Example**:

    | accessType | mmid | offset | tid | tmid | value |
    | ---------- | ---- | ------ | --- | ---- | ----- |
    | Read       | 0x01 | 0x10   | 5   | 0    | 0x42  |

- **Uniqueness Constraints**: Continuous write operations to the same address must be ordered chronologically. All `Init` type access operations must match the initial memory table $T_H$:  
  $\forall e \in T_M, \quad e.\text{type} = \text{Init} \implies (e.\text{address}, e.\text{value}) \in T_H$

##### 5) IO Circuits (Zero-Knowledge Support)

Since the WASM specification does not natively support zero-knowledge input and output (IO), ZAWA extends its IO circuit to support both **private inputs** and **public inputs**.

- **Public Inputs**: These are validated using polynomial lookup constraints to ensure that input values match the results of the `get_public_input` function in the execution circuit:  
  $\text{plookup}(T_{\text{public}}, (\text{inputCursor}, \text{publicInput})) = 0$
- **Private Inputs**: Stored in separate columns without explicit constraints, relying on zero-knowledge technology to hide the data.
- **Output Verification**: Output values must exist in a dedicated output column, and their correctness is verified using lookup constraints:  
  $\text{plookup}(T_{\text{output}}, (\text{auxCell}, \text{outputValue})) = 0$

---

### 1.3 Instruction Circuits\*\*

Once the architecture circuits are defined, the next step is to build corresponding circuits $C_{op}$ for each opcode (operation code) supported by the WASM specification. Since constraints are applied row by row in the execution trace circuit and each opcode's constraints may span multiple rows, we use **c.(curr + k)** to denote the cell in column **c** at row $curr + k$, which refers to the k-th row after the current row.

- **Row-wise Constraints Application**: The constraints for each instruction are activated through $\text{cur.start} \times (\text{cur.opcode} == op)$, and for each opcode $op_i$, the corresponding constraints $C_{op_i}$ are built and merged into the final constraint:

  $C_{op}(curr) := \sum_i \text{cur.start} \times (\text{cur.opcode} == op) \times C_{op_i}(\text{curr}) = 0$

- **Cross-row Interaction**: Data dependencies spanning across multiple rows are handled using **c.(curr + k)**.
- **Modular Verification**: Memory, stack, and frame table operations are verified using Plookup to ensure their validity.

### **Numeric Instructions (e.g., Add Instruction)**

Numeric instructions, such as addition, subtraction, and multiplication, are among the most common operations in WASM. **Constraints for these instructions include**:

- **Arithmetic Constraints**: The result must be computed correctly. For example, for the addition operation, the constraint is $\text{param0} + \text{param1} - \text{result} = 0$.
- **Stack Operation Validation**: Operands must come from valid stack accesses. The constraint is given by:

  $\text{plookup}(T_M, (\text{stack}, \text{read}, sp - k, tid, k, param_k)) = 0$

  ensuring that the value $param_k$ read from the stack position $sp - k$ is valid. The result must be written back to a valid stack location:

  $\text{plookup}(T_M, (\text{stack}, \text{write}, sp' - 1, tid, N, result)) = 0$

  ensuring that the result $result$ is written to the stack position $sp' - 1$.

- **Address Continuity**: The next instruction address is equal to the current address plus one:

  $iaddr_1 = iaddr_0 + 1$

- **Stack Pointer Update**: The new stack pointer is updated as:

  $sp' = sp - 1$

### **Control Flow Instructions**

Control flow instructions are used in WASM to alter the sequence of program execution. They primarily consist of three types: **fallthrough (default)**, **branch instructions**, and **call/return instructions**.

- **Call and Return Instructions**: The **call** instruction adds a new entry in the frame table $T_F$, pushes parameters onto the stack, and jumps to the target address. **Constraints** verify the validity of the parameters written to the stack, ensure the jump address is correct, and update the frame ID. The **return** instruction looks up the return address from the frame table and restores the previous frame state. **Constraints** ensure that the return address is correct, and the stack pointer and frame ID are correctly restored.

- **Branch Instructions**: Branch instructions, such as `br`, `br_if`, and conditional branches (`if-else`), calculate the jump address based on parameters and perform a direct jump. The **circuit constraint** ensures the validity of the parameters used in the jump and verifies that the next instruction's address is computed correctly based on the branch condition.

### **Memory/Stack/Global Instructions**

Memory, stack, and global variable instructions are used in WASM to manage data storage and access.

- **Instruction Composition**: Instructions are composed of (category, type, address, size, value), such as `(stack, read, sp, 64, w1)`.

- **Constraint Validation**: Read and write operations are validated using the memory access log circuit $T_M$. For example, for a read operation, the constraint is:

  $(\text{stack}, \text{read}, sp, tid, 0, w1) \in T_M$

  For write operations, constraints ensure that the written value is properly recorded, and subsequent reads retrieve the latest value.

### **Custom Instruction Extensions**

The goal of custom instructions is to optimize the circuit by reducing the total number of rows in the execution trace circuit $T_E$, thereby improving performance.

- **Inline Custom Instructions**: These are suitable for simple logic, such as bitwise operations, which can be compressed into a single instruction block. For example, a function like `sumLowest(x)` combines 4 bit extractions and 2 additions into a single constraint row.

- **External Function Proofs**: Complex logic, such as hash functions, can be validated using dedicated circuits that verify inputs and outputs. For example, in SHA256, custom instructions like `ch(x,y)` and `maj(x,y,z)` significantly reduce the number of rows.

---

### 1.4. Sharding and Batching\*\*

The Halo2 proof system has a **limit on the total number of rows** in the arithmetic circuit. When the scale of the WASM program grows too large, the full execution trace cannot fit into the circuit table (e.g., the $T_E$ table) all at once. The **solution** is to divide the execution trace into multiple sub-sequences (shards), generate independent proofs for each sub-sequence, and then merge them through **batching** to validate all sub-proofs. This approach **overcomes the row limit** and supports large-scale WASM program verification. The correctness of the full proof $P$ is guaranteed by the following conditions:

- **Sub-proof Verification**: Each sub-proof $P_k$ must be verified individually.
- **Glue Instructions**: A glue instruction is added at the end of each sub-sequence to enforce that its address is contiguous with the first instruction address of the next sub-sequence.
- **Log Continuity**: The constraints on the **Glue Instruction** table $T_{GM}$ and sub-log tables $T_M$ ensure continuity between sub-sequences.

#### Sharding and Batching Technical Details

- **Execution Trace Sharding**: The **sharding method** divides the full execution trace $[t_0, t_1, \dots, t_n]$ into multiple sub-blocks:  
  $t[a_0,b_0], t[a_1,b_1], \dots$, where each sub-block contains a sequence of contiguous instructions (for example, $t_0 - t_{100}$ is the first block, and $t_{101} - t_{200}$ is the second block). Each sub-block $t_{[a,b]}$ corresponds to independent memory access logs $M_{[a,b]}$, stack logs $SP_{[a,b]}$, and global variable logs $G_{[a,b]}$.

- **Sub-proof Generation**: A **sub-proof** $P_{[a,b]}$ is generated for each sub-block $t_{[a,b]}$ using the ZAWA circuit, verifying that the sub-block is valid within the context:
  $
  (F, M_{[a,b]}, SP_{[a,b]}, I(C,H), IO)
  $
  where $F$ is the frame table, $I$ is the initial data from the image, and $IO$ represents input/output. The goal is that the logical conjunction of all sub-proofs $P_{[0,k]}, P_{[k,2k]}, \dots$ is equivalent to the proof for the entire execution trace $P$.

#### Equivalence Constraints

- **Polynomial Constraints Equivalence**: The constraints at the end of each sub-block may differ from the overall circuit due to the shard boundaries. **Solution**: A glue instruction is inserted at the end of each sub-block to ensure address continuity with the first instruction of the next sub-block.  
  Example: The address of the glue instruction at the end of sub-block 1 is equal to the address of the first instruction of sub-block 2:

  $t_{b_{k+1}}.iaddr = t_{a_k}.iaddr$

- **Polynomial Lookup Equivalence**: The memory access log $T_M$ is split into sub-tables $T_M_1, T_M_2, \dots$, and these must satisfy global uniqueness constraints upon merging. **Solution**: Introduce a **glue table** $T_{GM}$ to track the boundary relationships between sub-tables. Constraints ensure that $T_{GM}$ and all $T_{M_k}$ tables satisfy global uniqueness constraints, ensuring the consistency of the memory access log across all sub-blocks.

#### Batch Processing Verification

- **Batch Processing Circuit**: The verification circuit for each sub-proof $P_k$ is integrated into the total batch processing circuit. Additional circuit modules check for address continuity, glue instructions, and log continuity across sub-sequences.

- **Optimization Directions**: Currently, verification logic is encoded into arithmetic circuits, but future research could explore more efficient batching solutions, such as recursive proofs or aggregation algorithms.

<details><summary><b> Code </b></summary>

<details><summary><b> zkwasm_circuit </b></summary>

```rust

// zkwasm_circuit
pub struct ZkWasmCircuitConfig<F: FieldExt> {
    shuffle_range_check_helper: (Column<Fixed>, Column<Fixed>, Column<Fixed>),
    rtable: RangeTableConfig<F>,
    image_table: ImageTableConfig<F>,
    post_image_table: PostImageTableConfig<F>,
    mtable: MemoryTableConfig<F>,
    frame_table: JumpTableConfig<F>,
    etable: EventTableConfig<F>,
    bit_table: BitTableConfig<F>,
    external_host_call_table: ExternalHostCallTableConfig<F>,
    context_helper_table: ContextContHelperTableConfig<F>,

    foreign_table_from_zero_index: Column<Fixed>,

    blinding_factors: usize,
}

```

</details>

<details><summary><b> etable </b></summary>

```rust
// etable
pub(crate) fn configure(
        meta: &mut ConstraintSystem<F>,
        (l_0, l_active, l_active_last): (Column<Fixed>, Column<Fixed>, Column<Fixed>),
        cols: &mut (impl Iterator<Item = Column<Advice>> + Clone),
        rtable: &RangeTableConfig<F>,
        image_table: &ImageTableConfig<F>,
        mtable: &MemoryTableConfig<F>,
        jtable: &JumpTableConfig<F>,
        bit_table: &BitTableConfig<F>,
        external_host_call_table: &ExternalHostCallTableConfig<F>,
        foreign_table_configs: &BTreeMap<&'static str, Box<dyn ForeignTableConfig<F>>>,
    ) -> EventTableConfig<F> {
        let step_sel = meta.fixed_column();

        let mut allocator = EventTableCellAllocator::new(
            meta,
            step_sel,
            (l_0, l_active, l_active_last),
            rtable,
            mtable,
            cols,
        );

        let ops = [0; OP_CAPABILITY].map(|_| allocator.alloc_bit_cell());
        let enabled_cell = allocator.alloc_bit_cell();

        let rest_mops_cell = allocator.alloc_common_range_cell();
        let rest_call_ops_cell = allocator.alloc_unlimited_cell();
        let rest_return_ops_cell = allocator.alloc_unlimited_cell();
        let input_index_cell = allocator.alloc_common_range_cell();
        let context_input_index_cell = allocator.alloc_common_range_cell();
        let context_output_index_cell = allocator.alloc_common_range_cell();
        let external_host_call_index_cell = allocator.alloc_common_range_cell();
        let sp_cell = allocator.alloc_common_range_cell();
        let mpages_cell = allocator.alloc_common_range_cell();
        let frame_id_cell = allocator.alloc_u32_state_cell();
        let eid_cell = allocator.alloc_u32_state_cell();
        let fid_cell = allocator.alloc_common_range_cell();
        let iid_cell = allocator.alloc_common_range_cell();
        let maximal_memory_pages_cell = allocator.alloc_common_range_cell();

        // We only need to enable equality for the cells of states
        let used_common_range_cells_for_state = allocator
            .free_cells
            .get(&EventTableCellType::CommonRange)
            .unwrap();
        allocator.enable_equality(
            meta,
            &EventTableCellType::CommonRange,
            used_common_range_cells_for_state.0
                + (used_common_range_cells_for_state.1 != 0) as usize,
        );

        let used_unlimited_cells_for_state = allocator
            .free_cells
            .get(&EventTableCellType::Unlimited)
            .unwrap();
        allocator.enable_equality(
            meta,
            &EventTableCellType::Unlimited,
            used_unlimited_cells_for_state.0 + (used_unlimited_cells_for_state.1 != 0) as usize,
        );

        let itable_lookup_cell = allocator.alloc_unlimited_cell();
        let brtable_lookup_cell = allocator.alloc_unlimited_cell();
        let jtable_lookup_cell = allocator.alloc_unlimited_cell();
        let is_returned_cell = allocator.alloc_bit_cell();
        let pow_table_lookup_modulus_cell = allocator.alloc_unlimited_cell();
        let pow_table_lookup_power_cell = allocator.alloc_unlimited_cell();
        let external_foreign_call_lookup_cell = allocator.alloc_unlimited_cell();
        let bit_table_lookup_cells = allocator.alloc_bit_table_lookup_cells();

        let mut foreign_table_reserved_lookup_cells = [(); FOREIGN_LOOKUP_CAPABILITY]
            .map(|_| allocator.alloc_unlimited_cell())
            .into_iter();

        let common_config = EventTableCommonConfig {
            enabled_cell,
            ops,
            rest_mops_cell,
            rest_call_ops_cell,
            rest_return_ops_cell,
            input_index_cell,
            context_input_index_cell,
            context_output_index_cell,
            external_host_call_index_cell,
            sp_cell,
            mpages_cell,
            frame_id_cell,
            eid_cell,
            fid_cell,
            iid_cell,
            maximal_memory_pages_cell,
            itable_lookup_cell,
            brtable_lookup_cell,
            jtable_lookup_cell,
            is_returned_cell,
            pow_table_lookup_modulus_cell,
            pow_table_lookup_power_cell,
            bit_table_lookup_cells,
            external_foreign_call_lookup_cell,
        };

        let mut op_bitmaps: BTreeMap<OpcodeClassPlain, usize> = BTreeMap::new();
        let mut op_configs: BTreeMap<OpcodeClassPlain, OpcodeConfig<F>> = BTreeMap::new();

        let mut profiler = AllocatorFreeCellsProfiler::new(&allocator);

        macro_rules! configure {
            ($op:expr, $x:ident) => {
                let op = OpcodeClassPlain($op as usize);

                let foreign_table_configs = BTreeMap::new();
                let mut constraint_builder = ConstraintBuilder::new(meta, &foreign_table_configs);

                let mut allocator = allocator.clone();
                let config = $x::configure(&common_config, &mut allocator, &mut constraint_builder);

                constraint_builder.finalize(|meta| {
                    (fixed_curr!(meta, step_sel), ops[op.index()].curr_expr(meta))
                });

                op_bitmaps.insert(op, op.index());
                op_configs.insert(op, OpcodeConfig::<F>(config));

                profiler.update(&allocator);
            };
        }

        configure!(OpcodeClass::BinShift, BinShiftConfigBuilder);
        configure!(OpcodeClass::Bin, BinConfigBuilder);
        configure!(OpcodeClass::BrIfEqz, BrIfEqzConfigBuilder);
        configure!(OpcodeClass::BrIf, BrIfConfigBuilder);
        configure!(OpcodeClass::Br, BrConfigBuilder);
        configure!(OpcodeClass::Call, CallConfigBuilder);
        configure!(OpcodeClass::CallHost, ExternalCallHostCircuitConfigBuilder);
        configure!(OpcodeClass::Const, ConstConfigBuilder);
        configure!(OpcodeClass::Conversion, ConversionConfigBuilder);
        configure!(OpcodeClass::Drop, DropConfigBuilder);
        configure!(OpcodeClass::GlobalGet, GlobalGetConfigBuilder);
        configure!(OpcodeClass::GlobalSet, GlobalSetConfigBuilder);
        configure!(OpcodeClass::LocalGet, LocalGetConfigBuilder);
        configure!(OpcodeClass::LocalSet, LocalSetConfigBuilder);
        configure!(OpcodeClass::LocalTee, LocalTeeConfigBuilder);
        configure!(OpcodeClass::Rel, RelConfigBuilder);
        configure!(OpcodeClass::Return, ReturnConfigBuilder);
        configure!(OpcodeClass::Select, SelectConfigBuilder);
        configure!(OpcodeClass::Test, TestConfigBuilder);
        configure!(OpcodeClass::Unary, UnaryConfigBuilder);
        configure!(OpcodeClass::Load, LoadConfigBuilder);
        configure!(OpcodeClass::Store, StoreConfigBuilder);
        configure!(OpcodeClass::BinBit, BinBitConfigBuilder);
        configure!(OpcodeClass::MemorySize, MemorySizeConfigBuilder);
        configure!(OpcodeClass::MemoryGrow, MemoryGrowConfigBuilder);
        configure!(OpcodeClass::BrTable, BrTableConfigBuilder);
        configure!(OpcodeClass::CallIndirect, CallIndirectConfigBuilder);

        macro_rules! configure_foreign {
            ($x:ident, $i:expr) => {
                let builder = $x::new($i);
                let op = OpcodeClass::ForeignPluginStart as usize + $i;
                let op = OpcodeClassPlain(op);

                let mut constraint_builder = ConstraintBuilder::new(meta, foreign_table_configs);
                let mut allocator = allocator.clone();

                let config = builder.configure(
                    &common_config,
                    &mut allocator,
                    &mut constraint_builder,
                    &mut foreign_table_reserved_lookup_cells,
                );

                constraint_builder.finalize(|meta| {
                    (fixed_curr!(meta, step_sel), ops[op.index()].curr_expr(meta))
                });

                op_bitmaps.insert(op, op.index());
                op_configs.insert(op, OpcodeConfig(config));

                profiler.update(&allocator);
            };
        }
        configure_foreign!(ETableWasmInputHelperTableConfigBuilder, 0);
        configure_foreign!(ETableContextHelperTableConfigBuilder, 1);
        configure_foreign!(ETableRequireHelperTableConfigBuilder, 2);

        profiler.assert_no_free_cells(&allocator);

        meta.create_gate("c1. enable seq", |meta| {
            vec![
                enabled_cell.next_expr(meta)
                    * (enabled_cell.curr_expr(meta) - constant_from!(1))
                    * fixed_curr!(meta, step_sel),
            ]
        });

        meta.create_gate("c4. opcode_bit lvl sum equals to 1", |meta| {
            vec![
                ops.map(|x| x.curr_expr(meta))
                    .into_iter()
                    .reduce(|acc, x| acc + x)
                    .unwrap()
                    - enabled_cell.curr_expr(meta),
            ]
            .into_iter()
            .map(|expr| expr * fixed_curr!(meta, step_sel))
            .collect::<Vec<_>>()
        });

        /*
         * How `* enabled_cell.curr_expr(meta)` effects on the separate step:
         *    1. constrains the relation between the last step and termination.
         *    2. ignores rows following the termination step.
         */
        let sum_ops_expr_with_init = |init: Expression<F>,
                                      meta: &mut VirtualCells<'_, F>,
                                      get_expr: &dyn Fn(
            &mut VirtualCells<'_, F>,
            &OpcodeConfig<F>,
        ) -> Option<Expression<F>>| {
            op_bitmaps
                .iter()
                .filter_map(|(op, op_index)| {
                    get_expr(meta, op_configs.get(op).unwrap())
                        .map(|expr| expr * ops[*op_index].curr_expr(meta))
                })
                .fold(init, |acc, x| acc + x)
                * fixed_curr!(meta, step_sel)
        };

        let sum_ops_expr = |meta: &mut VirtualCells<'_, F>,
                            get_expr: &dyn Fn(
            &mut VirtualCells<'_, F>,
            &OpcodeConfig<F>,
        ) -> Option<Expression<F>>| {
            op_bitmaps
                .iter()
                .filter_map(|(op, op_index)| {
                    get_expr(meta, op_configs.get(op).unwrap())
                        .map(|expr| expr * ops[*op_index].curr_expr(meta))
                })
                .reduce(|acc, x| acc + x)
                .unwrap()
        };

        meta.create_gate("c5a. rest_mops change", |meta| {
            vec![sum_ops_expr_with_init(
                rest_mops_cell.next_expr(meta) - rest_mops_cell.curr_expr(meta),
                meta,
                &|meta, config: &OpcodeConfig<F>| config.0.mops(meta),
            )]
        });

        meta.create_gate("c5b. rest jops change", |meta| {
            vec![
                sum_ops_expr_with_init(
                    rest_call_ops_cell.next_expr(meta) - rest_call_ops_cell.curr_expr(meta),
                    meta,
                    &|meta, config: &OpcodeConfig<F>| config.0.call_ops_expr(meta),
                ),
                sum_ops_expr_with_init(
                    rest_return_ops_cell.next_expr(meta) - rest_return_ops_cell.curr_expr(meta),
                    meta,
                    &|meta, config: &OpcodeConfig<F>| config.0.return_ops_expr(meta),
                ),
            ]
        });

        meta.create_gate("c5c. input_index change", |meta| {
            vec![sum_ops_expr_with_init(
                input_index_cell.curr_expr(meta) - input_index_cell.next_expr(meta),
                meta,
                &|meta, config: &OpcodeConfig<F>| {
                    config.0.input_index_increase(meta, &common_config)
                },
            )]
        });

        meta.create_gate("c5d. external_host_call_index change", |meta| {
            vec![sum_ops_expr_with_init(
                external_host_call_index_cell.curr_expr(meta)
                    - external_host_call_index_cell.next_expr(meta),
                meta,
                &|meta, config: &OpcodeConfig<F>| {
                    config
                        .0
                        .external_host_call_index_increase(meta, &common_config)
                },
            )]
        });

        meta.create_gate("c5e. sp change", |meta| {
            vec![sum_ops_expr_with_init(
                sp_cell.curr_expr(meta) - sp_cell.next_expr(meta),
                meta,
                &|meta, config: &OpcodeConfig<F>| config.0.sp_diff(meta),
            )]
        });

        meta.create_gate("c5f. mpages change", |meta| {
            vec![sum_ops_expr_with_init(
                mpages_cell.curr_expr(meta) - mpages_cell.next_expr(meta),
                meta,
                &|meta, config: &OpcodeConfig<F>| config.0.allocated_memory_pages_diff(meta),
            )]
        });

        meta.create_gate("c5g. context_input_index change", |meta| {
            vec![sum_ops_expr_with_init(
                context_input_index_cell.curr_expr(meta) - context_input_index_cell.next_expr(meta),
                meta,
                &|meta, config: &OpcodeConfig<F>| {
                    config.0.context_input_index_increase(meta, &common_config)
                },
            )]
        });

        meta.create_gate("c5h. context_output_index change", |meta| {
            vec![sum_ops_expr_with_init(
                context_output_index_cell.curr_expr(meta)
                    - context_output_index_cell.next_expr(meta),
                meta,
                &|meta, config: &OpcodeConfig<F>| {
                    config.0.context_output_index_increase(meta, &common_config)
                },
            )]
        });

        meta.create_gate("c6a. eid change", |meta| {
            vec![
                (eid_cell.next_expr(meta)
                    - eid_cell.curr_expr(meta)
                    - enabled_cell.curr_expr(meta))
                    * fixed_curr!(meta, step_sel),
            ]
        });

        meta.create_gate("c6b. fid change", |meta| {
            vec![sum_ops_expr_with_init(
                fid_cell.curr_expr(meta) - fid_cell.next_expr(meta),
                meta,
                &|meta, config: &OpcodeConfig<F>| {
                    config
                        .0
                        .next_fid(meta, &common_config)
                        .map(|x| x - fid_cell.curr_expr(meta))
                },
            )]
        });

        meta.create_gate("c6c. iid change", |meta| {
            vec![sum_ops_expr_with_init(
                iid_cell.next_expr(meta) - iid_cell.curr_expr(meta) - enabled_cell.curr_expr(meta),
                meta,
                &|meta, config: &OpcodeConfig<F>| {
                    config
                        .0
                        .next_iid(meta, &common_config)
                        .map(|x| iid_cell.curr_expr(meta) + enabled_cell.curr_expr(meta) - x)
                },
            )]
        });

        meta.create_gate("c6d. frame_id change", |meta| {
            vec![sum_ops_expr_with_init(
                frame_id_cell.curr_expr(meta) - frame_id_cell.next_expr(meta),
                meta,
                &|meta, config: &OpcodeConfig<F>| {
                    config
                        .0
                        .next_frame_id(meta, &common_config)
                        .map(|x| x - frame_id_cell.curr_expr(meta))
                },
            )]
        });

        meta.create_gate("c7. itable_lookup_encode", |meta| {
            let opcode = sum_ops_expr(meta, &|meta, config: &OpcodeConfig<F>| {
                Some(config.0.opcode(meta))
            });
            vec![
                (encode_instruction_table_entry(fid_cell.expr(meta), iid_cell.expr(meta), opcode)
                    - itable_lookup_cell.curr_expr(meta))
                    * enabled_cell.curr_expr(meta)
                    * fixed_curr!(meta, step_sel),
            ]
        });

        image_table.instruction_lookup(meta, "c8a. itable_lookup in itable", |meta| {
            itable_lookup_cell.curr_expr(meta) * fixed_curr!(meta, step_sel)
        });

        image_table.br_table_lookup(meta, "c8b. brtable_lookup in brtable", |meta| {
            brtable_lookup_cell.curr_expr(meta) * fixed_curr!(meta, step_sel)
        });

        jtable.configure_lookup_in_frame_table(meta, "c8c. jtable_lookup in jtable", |meta| {
            (
                fixed_curr!(meta, step_sel),
                common_config.is_returned_cell.curr_expr(meta) * fixed_curr!(meta, step_sel),
                common_config.jtable_lookup_cell.curr_expr(meta) * fixed_curr!(meta, step_sel),
            )
        });

        rtable.configure_in_pow_set(
            meta,
            "c8d. pow_table_lookup in pow_table",
            |meta| pow_table_lookup_power_cell.curr_expr(meta),
            |meta| pow_table_lookup_modulus_cell.curr_expr(meta),
            |meta| fixed_curr!(meta, step_sel),
        );

        external_host_call_table.configure_in_table(
            meta,
            "c8g. external_foreign_call_lookup in foreign table",
            |meta| {
                vec![
                    external_foreign_call_lookup_cell.curr_expr(meta) * fixed_curr!(meta, step_sel),
                ]
            },
        );

        bit_table.configure_in_table(meta, "c8f: bit_table_lookup in bit_table", |meta| {
            (
                fixed_curr!(meta, step_sel),
                fixed_curr!(meta, step_sel) * bit_table_lookup_cells.op.expr(meta),
                fixed_curr!(meta, step_sel) * bit_table_lookup_cells.left.expr(meta),
                fixed_curr!(meta, step_sel) * bit_table_lookup_cells.right.expr(meta),
                fixed_curr!(meta, step_sel) * bit_table_lookup_cells.result.expr(meta),
            )
        });

        meta.create_gate("c9. maximal memory pages consistent", |meta| {
            vec![
                (maximal_memory_pages_cell.next_expr(meta)
                    - maximal_memory_pages_cell.curr_expr(meta))
                    * fixed_curr!(meta, step_sel),
            ]
        });

        Self {
            step_sel,
            common_config,
            op_configs: Arc::new(op_configs),
        }
    }

```

</details>

<details><summary><b> mtable </b></summary>

```rust
// mtable
fn assign_entries(
        &self,
        region: &Region<'_, F>,
        mtable: &MemoryWritingTable,
        init_rest_mops: u64,
        _rest_memory_finalize_ops: u32,
    ) -> Result<(), Error> {
        macro_rules! assign_advice {
            ($ctx:expr, $cell:ident, $value:expr) => {
                self.config.$cell.assign($ctx, $value).unwrap()
            };
        }

        cfg_if::cfg_if! {
            if #[cfg(feature = "continuation")] {
                macro_rules! assign_u32_state {
                    ($ctx:expr, $cell:ident, $value:expr) => {
                        self.config.$cell.assign($ctx, $value).unwrap()
                    }
                }
            } else {
                macro_rules! assign_u32_state {
                    ($ctx:expr, $cell:ident, $value:expr) => {
                        assign_advice!($ctx, $cell, F::from($value as u64))
                    }
                }
            }
        }

        macro_rules! assign_bit {
            ($ctx:expr, $cell:ident) => {
                assign_advice!($ctx, $cell, F::one())
            };
        }

        macro_rules! assign_bit_if {
            ($ctx:expr, $cond:expr, $cell:ident) => {
                if $cond {
                    assign_advice!($ctx, $cell, F::one());
                }
            };
        }

        struct Status<F: FieldExt> {
            rest_mops: u64,

            init_encode: F,

            is_next_same_ltype_cell: bool,
            is_next_same_offset_cell: bool,
            offset_diff: u32,

            _rest_memory_finalize_ops: u32,
            _post_init_encode_cell: Option<F>,
        }

        let status = {
            let mut status = Vec::with_capacity(mtable.0.len());

            let mut rest_mops = init_rest_mops;
            let mut _rest_memory_finalize_ops = _rest_memory_finalize_ops;
            let mut current_address_init_encode = None;

            let mut iter = mtable.0.iter().peekable();

            let is_finalized_writing_entry =
                |entry: &MemoryWritingEntry, next_entry: Option<&&MemoryWritingEntry>| {
                    entry.entry.atype == AccessType::Write
                        && (next_entry.is_none()
                            || !next_entry
                                .as_ref()
                                .unwrap()
                                .entry
                                .is_same_location(&entry.entry))
                };

            while let Some(curr) = iter.next() {
                let next = iter.peek();

                if curr.entry.atype.is_init() {
                    current_address_init_encode =
                        Some(bn_to_field(&encode_init_memory_table_entry(
                            (curr.entry.ltype as u64).into(),
                            curr.entry.offset.into(),
                            (curr.entry.is_mutable as u64).into(),
                            curr.entry.eid.into(),
                            curr.entry.value.into(),
                        )));
                }

                let (is_next_same_ltype_cell, is_next_same_offset_cell, offset_diff) =
                    if let Some(next) = next {
                        if curr.entry.ltype == next.entry.ltype {
                            let offset_diff = next.entry.offset - curr.entry.offset;

                            (true, curr.entry.offset == next.entry.offset, offset_diff)
                        } else {
                            (false, false, 0u32)
                        }
                    } else {
                        (false, false, 0u32)
                    };

                status.push(Status {
                    rest_mops,

                    init_encode: current_address_init_encode.unwrap_or(F::zero()),

                    is_next_same_ltype_cell,
                    is_next_same_offset_cell,
                    offset_diff,

                    _rest_memory_finalize_ops,
                    _post_init_encode_cell: if is_finalized_writing_entry(curr, next) {
                        Some(bn_to_field(
                            &((encode_init_memory_table_address::<BigUint>(
                                (curr.entry.ltype as u64).into(),
                                curr.entry.offset.into(),
                            )) * MEMORY_ADDRESS_OFFSET
                                + (encode_init_memory_table_entry::<BigUint>(
                                    (curr.entry.ltype as u64).into(),
                                    curr.entry.offset.into(),
                                    (curr.entry.is_mutable as u64).into(),
                                    curr.entry.eid.into(),
                                    curr.entry.value.into(),
                                ))),
                        ))
                    } else {
                        None
                    },
                });

                if let Some(next_entry) = next {
                    if !next_entry.entry.is_same_location(&curr.entry) {
                        current_address_init_encode = None;
                    }
                }

                if is_finalized_writing_entry(curr, next) {
                    _rest_memory_finalize_ops -= 1;
                }

                if !curr.entry.atype.is_init() {
                    rest_mops -= 1;
                }
            }

            status
        };

        const THREAD: usize = 8;
        let chunk_size = if mtable.0.is_empty() {
            1
        } else {
            (mtable.0.len() + THREAD - 1) / THREAD
        };

        mtable
            .0
            .par_chunks(chunk_size)
            .enumerate()
            .for_each(|(chunk_index, entries)| {
                let mut ctx = Context::new(region);
                ctx.offset = (chunk_index * chunk_size) * MEMORY_TABLE_ENTRY_ROWS as usize;
                let mut invert_cache: HashMap<u64, F> = HashMap::default();

                for (index, entry) in entries.iter().enumerate() {
                    let index = chunk_index * chunk_size + index;

                    assign_bit!(&mut ctx, enabled_cell);

                    match entry.entry.ltype {
                        LocationType::Stack => assign_bit!(&mut ctx, is_stack_cell),
                        LocationType::Heap => assign_bit!(&mut ctx, is_heap_cell),
                        LocationType::Global => assign_bit!(&mut ctx, is_global_cell),
                    };

                    assign_bit_if!(&mut ctx, entry.entry.is_mutable, is_mutable);

                    match entry.entry.vtype {
                        VarType::I32 => assign_bit!(&mut ctx, is_i32_cell),
                        VarType::I64 => assign_bit!(&mut ctx, is_i64_cell),
                    };

                    assign_bit_if!(&mut ctx, entry.entry.atype.is_init(), is_init_cell);

                    assign_u32_state!(&mut ctx, start_eid_cell, entry.entry.eid);
                    assign_u32_state!(&mut ctx, end_eid_cell, entry.end_eid);
                    assign_u32_state!(&mut ctx, eid_diff_cell, entry.end_eid - entry.entry.eid - 1);
                    assign_advice!(&mut ctx, init_encode_cell, status[index].init_encode);
                    assign_advice!(&mut ctx, rest_mops_cell, F::from(status[index].rest_mops));
                    assign_advice!(&mut ctx, offset_cell, entry.entry.offset);
                    assign_advice!(&mut ctx, value, entry.entry.value);

                    let offset_diff = F::from(status[index].offset_diff as u64);
                    let offset_diff_inv = invert_cache
                        .entry(status[index].offset_diff as u64)
                        .or_insert_with(|| offset_diff.invert().unwrap_or(F::zero()));
                    let offset_diff_inv_helper = if status[index].offset_diff == 0 {
                        F::zero()
                    } else {
                        F::one()
                    };

                    assign_bit_if!(
                        &mut ctx,
                        status[index].is_next_same_ltype_cell,
                        is_next_same_ltype_cell
                    );
                    assign_bit_if!(
                        &mut ctx,
                        status[index].is_next_same_offset_cell,
                        is_next_same_offset_cell
                    );
                    assign_advice!(&mut ctx, offset_diff_cell, status[index].offset_diff);
                    assign_advice!(&mut ctx, offset_diff_inv_cell, *offset_diff_inv);
                    assign_advice!(
                        &mut ctx,
                        offset_diff_inv_helper_cell,
                        offset_diff_inv_helper
                    );

                    #[cfg(feature = "continuation")]
                    {
                        assign_advice!(
                            &mut ctx,
                            rest_memory_finalize_ops_cell,
                            F::from(status[index]._rest_memory_finalize_ops as u64)
                        );

                        assign_advice!(
                            &mut ctx,
                            address_encode_cell,
                            bn_to_field(&encode_init_memory_table_address(
                                (entry.entry.ltype as u64).into(),
                                entry.entry.offset.into()
                            ))
                        );

                        if let Some(post_init_encode) = status[index]._post_init_encode_cell {
                            assign_advice!(&mut ctx, post_init_encode_cell, post_init_encode);
                        }
                    }

                    assign_advice!(
                        &mut ctx,
                        encode_cell,
                        bn_to_field(&encode_memory_table_entry(
                            entry.entry.offset.into(),
                            (entry.entry.ltype as u64).into(),
                            if VarType::I32 == entry.entry.vtype {
                                1u64.into()
                            } else {
                                0u64.into()
                            }
                        ))
                    );

                    ctx.step(MEMORY_TABLE_ENTRY_ROWS as usize);
                }
            });

        Ok(())
    }

```

</details>

<details><summary><b> jtable </b></summary>

```rust

// jtable
    fn configure_rest_jops_decrease(&self, meta: &mut ConstraintSystem<F>) {
        /*
         * Why we do not need `enable == 1 -> encode != 0`.
         *   If enable == 1 but encode == 0, it means the number of ops may greater than the number of encoding. However
         *   - If the number of ops is not correct, the equality between etable and frame table will fail.
         *   - If the number of ops is correct, encode == 0 implies an entry is missing and etable cannot
         *     lookup the correct entry in frame table.
         */
        meta.create_gate("c3. jtable rest decrease", |meta| {
            vec![
                (self.rest_return_ops(meta)
                    - self.next_rest_return_ops(meta)
                    - self.returned(meta) * self.enable(meta))
                    * self.sel(meta),
                (self.rest_call_ops(meta) - self.next_rest_call_ops(meta) - self.enable(meta)
                    + self.inherited_bit(meta) * self.enable(meta))
                    * self.sel(meta),
            ]
        });
    }

```

</details>

</details>

---

## 2. SP1: A Zero-Knowledge Virtual Machine (zkVM) for RISC-V\*\*

SP1 is a **Zero-Knowledge Virtual Machine (zkVM)** developed and maintained by **Succinct**. It is designed to generate **zero-knowledge proofs** for RISC-V programs. Developers can write code in Rust, specify inputs and outputs, and use SP1 to generate a proof that validates the correctness of the output. The proof generated follows the **Scalable Transparent Argument of Knowledge (STARK)** format.

SP1 is built on **plonky3**, an open-source modular proof system toolkit developed by the **Polygon Zero** team. The system leverages standard **STARK** technology, which includes **Algebraic Intermediate Representations (AIR)** for arithmetic program logic, **polynomial commitments** based on batch-encoded Reed-Solomon coding, and **cross-table lookups** based on the LogUp lookup protocol.

The operations in SP1 are based on the **Baby Bear finite field** 𝔽 (with prime order \(15 \times 2^{27} + 1\)) and its 4th-degree extension field 𝔽(4).

**SP1 Workflow**

- **Rust Program Compilation**: The user’s Rust code is compiled into RISC-V assembly code. RISC-V is a standard instruction set architecture (ISA), and Rust has native support for targeting this architecture.
- **Execution Record Generation**: During runtime, SP1 processes the RISC-V ELF (Executable and Linkable Format) into an **ExecutionRecord**, which is a list of executed instructions and memory states.
- **Execution Trace Generation**: The **ExecutionRecord** is then used to generate the **execution trace**, a table of field 𝔽 elements accompanied by polynomial constraints that ensure the correctness of the execution. The prover’s task is to demonstrate that these constraints are satisfied.
- **Multi-table Coordination and Sharding**: In practice, the execution trace is divided into multiple tables, each with its own set of constraints. These tables are coordinated using the LogUp protocol. When a single proof becomes too large, SP1 shards the tables into **shards**, generates individual proofs for each shard, and then coordinates them via LogUp.
- **Recursive Proofs**: To generate constant-sized proofs, Succinct has designed a specialized recursive virtual machine, domain-specific language (DSL), and a custom instruction set architecture (ISA).

---

### 2.1 SP1 Multi-table Architecture\*\*

SP1 employs a **multi-table collaborative architecture** to generate execution proofs. Each table has a specific role, and they are coordinated via the **LogUp protocol**. The core modules include:

1. **CPU Table**: This table records the execution trace of RISC-V instructions, with each row corresponding to a single clock cycle.

   - **Fields**: `pc` (program counter), pointing to the current instruction address; `clk` (clock cycle counter); `a, b, c` (the input/output operands of instructions).
   - **Constraints**: Verifies the continuity of `pc` and `clk` (e.g., for jump instructions, the next instruction address is validated). **Operand sources** are not directly verified in this table but are validated through lookups in other tables (e.g., memory tables).

2. **ALU Table**: This table verifies the correctness of arithmetic and bitwise operations (addition, subtraction, multiplication, division, shifts, etc.). It receives lookup requests from the CPU table and validates the operands `b, c` and the result `a`.

3. **Memory Tables**: The memory is divided into:

   - **MemoryInit**: Records the initial memory state of the program.
   - **MemoryFinal**: Records the final memory state at program termination.
   - **ProgramMemory**: Preprocessed constants in the program’s memory.

   **Optimization**: Instead of recording the full memory state, only the “diff” (differences) in memory access is logged to reduce storage overhead.

4. **Precompiled Tables**: These are used for efficient processing of cryptographic operations (e.g., hashing, elliptic curve operations), preventing complex calculations from being expanded in the CPU table. For example, the **SHA256/Keccak** hash functions and elliptic curve operations (point addition, doubling, coordinate decompression for curves like secp256k1, ed25519) are handled here. The **ecall** instruction dispatches these operations to the precompiled table, and the CPU table verifies the results via lookups.
5. **Other Tables**:
   - **Bytes Table**: Handles arithmetic operations and range checks on `u8` values.
   - **Program Table**: Preprocessed program code segments, with the CPU table fetching instructions based on the program counter `pc`.

#### **Inter-table Communication via LogUp Protocol and Memory Consistency Verification**

**LogUp Protocol**: This protocol is used to verify row consistency across tables (e.g., matching operations between the CPU and ALU tables).

- **Process**:

  - Initially, **fingerprints** are sent/received: the sender generates a "send fingerprint" and the receiver generates a "receive fingerprint."
  - **Bus accumulation**: All fingerprints are accumulated into a cross-table bus.
  - **Final validation**: At the end of the proof, the total sum of the sent and received fingerprints is verified for consistency.

- **Column Extensions**:
  - **Permutation Columns**: Track the order of lookup requests.
  - **Accumulator Columns**: Track the accumulated sum of fingerprints.
    The fingerprint calculations are done in the extended field 𝔽(4) to enhance security.

**Memory Consistency Verification**:

- **Virtual Memory Tracking**: Memory access is recorded as a read operation followed by a write operation (which may write the same value).
- **Set Matching**: Maintain read set `R` and write set `W` and use LogUp to verify that `R = W`, ensuring correct operation sequencing.
- **Timestamp Constraints**: Ensure that each read operation’s timestamp `t` is earlier than the corresponding write operation’s timestamp `t'`.

#### **Sharding and Recursive Proofs**

**Sharding**: SP1 can handle large-scale programs (e.g., Tendermint light client verification, which involves about 30 million cycles) by splitting the execution trace into multiple shards, each with a maximum height of \(2^{22}\) (around 2 million rows).

- **Independent Proofs**: For each shard, an individual STARK proof is generated.
- **Cross-shard Coordination**: The LogUp protocol is used to ensure the consistency of interactions between shards.

**Recursive Proofs**: These are used to generate a constant-sized final proof, suitable for on-chain verification.

- **Sharded Proof Tree**: Each shard proof is treated as a leaf node, and recursive merging forms higher-level nodes.
- **Custom Compiler**: The recursive verification logic is compiled into custom assembly instructions optimized for the BabyBear field.
- **GROTH16 Wrapping**: The root node proof is converted into the GROTH16 format, improving on-chain verification efficiency.

### 2.2 Core Protocols Used by SP1

**FRI**, **AIR**, and **LogUp** are the key protocols used in SP1 to ensure the integrity and consistency of multi-table interactions and to validate the correctness of the computation.

#### 1) **FRI Protocol**

FRI (Fast Reed-Solomon Interactive Oracle Proof of Proximity) is a fundamental protocol employed in SP1 to verify the low-degree polynomial properties of a function within a given domain. The core idea behind FRI is to iteratively "fold" the problem to progressively reduce its size, ultimately verifying whether a very low-degree polynomial satisfies specific conditions. FRI is essential to the STARK proof system, as it efficiently verifies polynomial properties.

**FRI Parameters**

- **Field**: Baby Bear Field 𝔽, of order $15 × 2^{27} + 1$.
- **Expansion Factor**: 2 (Reed-Solomon code rate is 1/2).
- **Folding Method**: Binomial folding (each iteration halves the field size).
- **Batch Processing**: Optimized for multi-round folding.

**How FRI Works**

- **Initial Problem**: Given a domain K, a subset $𝑆 ⊆ K (with size 𝑛)$, and degree d < n, FRI is used to prove that a function $𝑓: 𝑆 → K$ is a polynomial of degree less than d.
- **Folding Process**: In each round, FRI folds f into a new function f′, defined over a subset 𝑆′ of size n/2, and the degree of f′ is reduced to d/2. If f is not a low-degree polynomial, f′ is unlikely to be one either.
- **Final Validation**: After approximately log(𝑑) folding rounds, the verifier checks whether the final function is constant. If it is constant, the proof is accepted; otherwise, it is rejected.

#### 2) **AIR (Algebraic Intermediate Representation)**

SP1 uses **AIR** to translate the program’s execution into polynomial constraints, ensuring the correctness of the computation. AIR allows complex computational processes to be expressed as polynomial constraints, which are then verified using FRI, ensuring that the computations are accurate and consistent.

**Example: Fibonacci Sequence**

Consider the task of verifying that a program correctly computes the 5th Fibonacci number 𝐹₅ = 8. AIR would apply the following polynomial constraints:

- **Boundary Conditions**:
  - $𝑋_0 − 1 = 0$
  - $𝑋_1 − 1 = 0$
- **Transition Conditions**:
  - $𝑋_{𝑖+2} − 𝑋_{𝑖+1} − 𝑋_𝑖 = 0 for 𝑖 ∈ {0, 1, 2, 3}$

These constraints ensure that the sequence $x_0, ..., x_5$ adheres to the Fibonacci sequence definition.

**Polynomial Transformation**

1. **Multi-variable to Single-variable**: Choose a generator 𝑜 for the field 𝕂 and construct a univariate polynomial 𝑝 such that $𝑝(𝑜^i) = 𝑥ᵢ$.
2. **Polynomial Constraints**: Convert the boundary and transition conditions into a single-variable polynomial $f(𝑥) = p(o^2𝑥) − p(0𝑥) − p(x)$ and verify that its values in the specific domain are zero.
3. **Low-degree Polynomial Verification**: Define the quotient polynomial as $f(x) = \frac{p(o^2x) − p(ox) − p(x)}{(x−1)(x−o)(x−o^2)(x−o^3)}$, and use FRI to validate the low-degree nature of f(x).

#### 3) **LogUp Lookup Protocol**

The **LogUp** (Log Derivative Lookup Argument) protocol is used to ensure consistency between multiple tables in SP1. It works by constructing polynomial equalities through logarithmic derivatives, allowing verification that the rows of different tables maintain a specific relationship. LogUp is essential for coordinating multi-table interactions in SP1.

**Example: CPU Table and ADD Table**

Consider two tables: the **CPU Table**, which records the execution results of each instruction, and the **ADD Table**, which records the results of addition instructions. The LogUp protocol’s goal is to verify that the rows in the ADD Table are some permutation of the rows of addition operations in the CPU Table.

| **Table 1: CPU Table** | **Table 2: ADD Table** |
| ---------------------- | ---------------------- |
| Cycle 0: ADD 3 (2+1)   | Cycle 2: ADD 4 (2+2)   |
| Cycle 1: MUL 6 (2×3)   | Cycle 0: ADD 3 (2+1)   |
| Cycle 2: ADD 4 (2+2)   |                        |

**Lemma**: For two matrices with fewer rows than the field’s characteristic p, the rows of the two matrices are permutations of each other if and only if the following condition holds:

$\sum_{i=0}^{n-1} \frac{1}{X + \sum_{j} Y^j A_{ij}} = \sum_{i=0}^{n-1} \frac{1}{X + \sum_{j} Y^j B_{ij}}$

**Intuitive Explanation**:

- The left-hand side is the "inverse sum" of all rows in the CPU Table, and the right-hand side is the inverse sum of rows in the ADD Table.
- If the rows are permutations of each other, their inverse sums will be equal for any values of X and Y.
- This equation essentially serves as a consistency check for the "row hash" between the two tables.

**LogUp Protocol Steps**

- **Commitment**: The prover commits to the row data in the tables.
- **Random Sampling**: The verifier sends random numbers 𝛾 and 𝛽 from the field 𝕂.
- **Computation**: The prover computes and sends 𝑠ᵢ and 𝑟ᵢ (representing the inverse elements for the ADD and CPU tables, respectively):
  - CPU Table: $s_i = \frac{1}{\gamma + \sum \beta^j A_{ij}}$
    (For example, Cycle 0: $1/(1 + 2^1×2 + 2^0×1) = 1/6 \mod 7 = 6$)
  - ADD Table: $r_i = \frac{m_i}{\gamma + \sum \beta^j B_{ij}}$
    (For example, Cycle 2: $1/(1 + 2^1×2 + 2^0×2) = 1/7 \mod 7 = 1$)
- **Verification**: The verifier checks if the computed values $𝑠_i$ and $𝑟_i$ satisfy a polynomial equality, and if the sum of $𝑟_i$ equals the sum of $s_i$.
- **Running Sum Column**: To optimize the verification, a "Running Sum Column" is used. The final row of the Running Sum is verified to ensure the overall consistency of the inverse sums.

These core protocols—**FRI**, **AIR**, and **LogUp**—work together to ensure the efficiency, correctness, and security of SP1, making it an effective tool for generating zkSNARK proofs in a highly scalable manner.

<details><summary><b> Code </b></summary>

<details><summary><b> add_sub </b></summary>

```rust
  // add_sub
  fn eval(&self, builder: &mut AB) {
          let main = builder.main();
          let local = main.row_slice(0);
          let local: &AddSubCols<AB::Var> = (*local).borrow();

          // SAFETY: All selectors `is_add` and `is_sub` are checked to be boolean.
          // Each "real" row has exactly one selector turned on, as `is_real = is_add + is_sub` is boolean.
          // Therefore, the `opcode` matches the corresponding opcode of the instruction.
          let is_real = local.is_add + local.is_sub;
          builder.assert_bool(local.is_add);
          builder.assert_bool(local.is_sub);
          builder.assert_bool(is_real.clone());

          let opcode = AB::Expr::from_f(Opcode::ADD.as_field()) * local.is_add
              + AB::Expr::from_f(Opcode::SUB.as_field()) * local.is_sub;

          // Evaluate the addition operation.
          // This is enforced only when `op_a_not_0 == 1`.
          // `op_a_val` doesn't need to be constrained when `op_a_not_0 == 0`.
          AddOperation::<AB::F>::eval(
              builder,
              local.operand_1,
              local.operand_2,
              local.add_operation,
              local.op_a_not_0.into(),
          );

          // SAFETY: We check that a padding row has `op_a_not_0 == 0`, to prevent a padding row sending byte lookups.
          builder.when(local.op_a_not_0).assert_one(is_real.clone());

          // Receive the arguments.  There are separate receives for ADD and SUB.
          // For add, `add_operation.value` is `a`, `operand_1` is `b`, and `operand_2` is `c`.
          // SAFETY: This checks the following. Note that in this case `opcode = Opcode::ADD`
          // - `next_pc = pc + 4`
          // - `num_extra_cycles = 0`
          // - `op_a_val` is constrained by the `AddOperation` when `op_a_not_0 == 1`
          // - `op_a_not_0` is correct, due to the sent `op_a_0` being equal to `1 - op_a_not_0`
          // - `op_a_immutable = 0`
          // - `is_memory = 0`
          // - `is_syscall = 0`
          // - `is_halt = 0`
          builder.receive_instruction(
              AB::Expr::zero(),
              AB::Expr::zero(),
              local.pc,
              local.pc + AB::Expr::from_canonical_u32(DEFAULT_PC_INC),
              AB::Expr::zero(),
              opcode.clone(),
              local.add_operation.value,
              local.operand_1,
              local.operand_2,
              AB::Expr::one() - local.op_a_not_0,
              AB::Expr::zero(),
              AB::Expr::zero(),
              AB::Expr::zero(),
              AB::Expr::zero(),
              local.is_add,
          );

          // For sub, `operand_1` is `a`, `add_operation.value` is `b`, and `operand_2` is `c`.
          // SAFETY: This checks the following. Note that in this case `opcode = Opcode::SUB`
          // - `next_pc = pc + 4`
          // - `num_extra_cycles = 0`
          // - `op_a_val` is constrained by the `AddOperation` when `op_a_not_0 == 1`
          // - `op_a_not_0` is correct, due to the sent `op_a_0` being equal to `1 - op_a_not_0`
          // - `op_a_immutable = 0`
          // - `is_memory = 0`
          // - `is_syscall = 0`
          // - `is_halt = 0`
          builder.receive_instruction(
              AB::Expr::zero(),
              AB::Expr::zero(),
              local.pc,
              local.pc + AB::Expr::from_canonical_u32(DEFAULT_PC_INC),
              AB::Expr::zero(),
              opcode,
              local.operand_1,
              local.add_operation.value,
              local.operand_2,
              AB::Expr::one() - local.op_a_not_0,
              AB::Expr::zero(),
              AB::Expr::zero(),
              AB::Expr::zero(),
              AB::Expr::zero(),
              local.is_sub,
          );
      }
```

</details>

<details><summary><b> cpu </b></summary>

```rust
      // cpu

      fn eval(&self, builder: &mut AB) {
          let main = builder.main();
          let (local, next) = (main.row_slice(0), main.row_slice(1));
          let local: &CpuCols<AB::Var> = (*local).borrow();
          let next: &CpuCols<AB::Var> = (*next).borrow();

          let public_values_slice: [AB::PublicVar; SP1_PROOF_NUM_PV_ELTS] =
              core::array::from_fn(|i| builder.public_values()[i]);
          let public_values: &PublicValues<Word<AB::PublicVar>, AB::PublicVar> =
              public_values_slice.as_slice().borrow();

          // We represent the `clk` with a 16 bit limb and a 8 bit limb.
          // The range checks for these limbs are done in `eval_shard_clk`.
          let clk =
              AB::Expr::from_canonical_u32(1u32 << 16) * local.clk_8bit_limb + local.clk_16bit_limb;

          // Program constraints.
          // SAFETY: `local.is_real` is checked to be boolean in `eval_is_real`.
          // The `pc` and `instruction` is taken from the `ProgramChip`, where these are preprocessed.
          builder.send_program(local.pc, local.instruction, local.is_real);

          // Register constraints.
          self.eval_registers::<AB>(builder, local, clk.clone());

      ...

      /// Constraints related to the shard and clk.
      ///
      /// This method ensures that all of the shard values are the same and that the clk starts at 0
      /// and is transitioned appropriately.  It will also check that shard values are within 16 bits
      /// and clk values are within 24 bits.  Those range checks are needed for the memory access
      /// timestamp check, which assumes those values are within 2^24.  See
      /// [`MemoryAirBuilder::verify_mem_access_ts`].
      pub(crate) fn eval_shard_clk<AB: SP1AirBuilder>(
          &self,
          builder: &mut AB,
          local: &CpuCols<AB::Var>,
          next: &CpuCols<AB::Var>,
          public_values: &PublicValues<Word<AB::PublicVar>, AB::PublicVar>,
          clk: AB::Expr,
      ) {

      ...

      /// Constraints related to the pc for non jump, branch, and halt instructions.
      ///
      /// The function will verify that the pc increments by 4 for all instructions except branch,
      /// jump and halt instructions. Also, it ensures that the pc is carried down to the last row
      /// for non-real rows.
      pub(crate) fn eval_pc<AB: SP1AirBuilder>(
          &self,
          builder: &mut AB,
          local: &CpuCols<AB::Var>,
          next: &CpuCols<AB::Var>,
          public_values: &PublicValues<Word<AB::PublicVar>, AB::PublicVar>,
      ) {

      ...

      /// Constraints related to the is_real column.
      ///
      /// This method checks that the is_real column is a boolean.  It also checks that the first row
      /// is 1 and once its 0, it never changes value.
      pub(crate) fn eval_is_real<AB: SP1AirBuilder>(
          &self,
          builder: &mut AB,
          local: &CpuCols<AB::Var>,
          next: &CpuCols<AB::Var>,
      ) {

```

</details>

<details><summary><b> memory </b></summary>

```rust
  // memory
  impl<F: PrimeField32> MachineAir<F> for MemoryLocalChip {
      fn generate_trace(
          &self,
          input: &ExecutionRecord,
          _output: &mut ExecutionRecord,
      ) -> RowMajorMatrix<F> {
      }

  ...

      fn eval(&self, builder: &mut AB) {
      }
  }

  ...

  // memory/global.rs
  impl<F: PrimeField32> MachineAir<F> for MemoryGlobalChip {
      fn generate_trace(
          &self,
          input: &ExecutionRecord,
          _output: &mut ExecutionRecord,
      ) -> RowMajorMatrix<F> {
      }

      fn eval(&self, builder: &mut AB) {
      }
  }
```

 </details>

<details><summary><b> core & contraints </b></summary>

```rust
  // core.r
  /// A program for recursively verifying a batch of SP1 proofs.
  #[derive(Debug, Clone, Copy)]
  pub struct SP1RecursiveVerifier<C, SC> {
      _config: PhantomData<(C, SC)>,
  }

  ...

  pub struct SP1RecursionWitnessVariable<C: CircuitConfig<F = BabyBear>, SC: BabyBearFriConfigVariable<C>> {
  }

  // constraints.rs
  pub fn verify_constraints(
      builder: &mut Builder<C>,
      chip: &MachineChip<SC, A>,
      opening: &ChipOpenedValues<Felt<C::F>, Ext<C::F, C::EF>>,
      trace_domain: TwoAdicMultiplicativeCoset<C::F>,
  ) {
      let sels = trace_domain.selectors_at_point_variable(builder, zeta);
  }

```

</details>

</details>

---

[zkWasm-main](https://github.com/ETAAcademy/ETAAcademy-ZK-Meme/tree/main/Appendix/zkWasm-main)

[sp1-dev](https://github.com/ETAAcademy/ETAAcademy-ZK-Meme/tree/main/Appendix/sp1-dev)

<div  align="center"> 
<img src="images/53_ZKVM_Circuit.gif" width="50%" />
</div>
