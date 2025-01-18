# ETAAcademy-ZKMeme: 52. ISA, SP1 & RISCV

<table>
  <tr>
    <th>title</th>
    <th>tags</th>
  </tr>
  <tr>
    <td>52. ISA, SP1 & RISCV</td>
    <td>
      <table>
        <tr>
          <th>zk-meme</th>
          <th>basic</th>
          <th>quick_read</th>
          <td>52. ISA, SP1 & RISCV</td>
        </tr>
      </table>
    </td>
  </tr>
</table>

[Github](https://github.com/ETAAcademy)｜[Twitter](https://twitter.com/ETAAcademy)｜[ETA-ZK-Meme](https://github.com/ETAAcademy/ETAAcademy-ZK-Meme)

Authors: [Evta](https://twitter.com/pwhattie), looking forward to your joining

# Optimizing zkVM Design: Leveraging ISA, SP1, and RISC-V for Efficient Zero-Knowledge Proofs

A **virtual machine (VM)** is a software-based simulation or virtualization of a computer system. It provides functionalities similar to a physical computer, enabling the execution of operating systems (OS) or applications. A **zkVM** (zero-knowledge virtual machine) extends the concept of a traditional VM by integrating **zero-knowledge proof capabilities**. While executing code, a zkVM simultaneously generates a zero-knowledge proof that attests to the correctness of the computation.

In zkVM, the compiler translates high-level programming languages into low-level **assembly language**, which closely mirrors machine instructions and directs the hardware to perform tasks.

#### The Role of Instruction Set Architecture (ISA)

In computer science, **Instruction Set Architecture (ISA)** serves as an abstract model defining how software interacts with the CPU or a computing system. The devices or programs (e.g., CPUs) that implement a given ISA act as its embodiment. ISA bridges the gap between programs and the underlying machine, including virtual machines.

In zkVM design, the choice of ISA is crucial. Adopting mainstream ISAs such as **RISC-V** or **MIPS** ensures broader language support and access to robust development tools. Engineers can use familiar high-level languages like **Rust** or **C++** to develop for these platforms.

A zkVM’s design must prioritize **performance**, focusing on hardware acceleration, efficient memory management, and finite field optimizations. The **streamlined instruction set design** promotes better execution efficiency, simplified implementation, and higher compatibility with the high-performance demands of zero-knowledge proofs.

- **Hardware Acceleration**: Supporting architectures like **SIMD (Single Instruction, Multiple Data)** and **GPU (Graphics Processing Unit)** can significantly boost computational throughput.

- **Data Optimization**: Favoring 32-bit data types, such as **FP32**, enhances GPU and SIMD operations since **FP32 outperforms FP64** in computational efficiency.

- **Memory Management**: Optimizing memory address operations (`usize`) and caching strategies ensures that the zkVM can handle large-scale computations efficiently.

- **Finite Field Selection**: Smaller finite fields (less than 32 bits) improve computational efficiency in zero-knowledge proof generation.

Unlike traditional VMs that simulate a "real" computer, zkVMs employ **high-level abstractions** to design a lightweight and efficient architecture. This approach aligns with the needs of high-speed execution and verification in zkVMs, resembling a **finite-state machine (FSM)** model.

#### zkVM as a Finite-State Machine

A zkVM operates as a **finite-state machine (FSM)**, enabling lightweight and efficient processing. Its key components include:

- **State**: Represented by **memory** and **registers**, encapsulating data and context during program execution.

- **Actions**: Each instruction is an action that updates the system’s state.

- **Instruction Cycle**: Instructions are fetched from memory, decoded, and executed in sequence.

This simplified architecture aligns with zkVM's goal of high performance and efficiency.

#### Precompilation and Recursive Proving

The combination of **precompilation** and **recursive proving** empowers zkVMs with exceptional performance advantages. Precompilation optimizes common operations, such as elliptic curve computations and hash calculations, through two main methods:

- **Custom Instructions**: For instance, Rust's `asm!` macro allows integration of custom assembly instructions directly into libraries. However, this approach may face compatibility issues with third-party VMs and toolchains.
- **Custom System Calls (ABI Layer)**: This method maintains ISA compliance while introducing new target tiers, such as a `zkvm` target in Rust, to define ABI specifications for zkVMs.

**Recursive proof aggregation** further enhances performance by compressing multiple proofs into one through recursive verification, improving efficiency and reducing proof size.

#### Extended Functionality

- **Input and Output (IO)**: zkVMs support **private inputs** (witnesses) and **public outputs** (shared values). Standard IO features like **stdin**, **stdout**, and **stderr** facilitate debugging and data handling.

- **Debugging Tools**: Functions like `Debug` and `Trace` enable execution tracking, aiding in problem localization and resolution.

- **Memory Paging**: Memory pagers optimize **Merkle proofs**, reducing storage proof overhead and enhancing the efficiency of read/write operations.

---

### Memory Management Unit (MMU) for Flexible Memory Management in Emulators

Managing memory access efficiently is critical for any emulator. This process involves implementing a **Memory Management Unit (MMU)** to handle various memory regions, which may not only store data but also trigger special operations or return dynamic values upon access. To achieve this, an **Addressable Interface** is introduced, providing a unified way to manage memory reads, writes, and address range checks.

The **Addressable Interface** includes three core methods:

- **`Contains`**: Checks if a specified address belongs to a memory region.
- **`Read`**: Reads data from a specified address.
- **`Write`**: Writes data to a specified address.

By implementing this interface, each memory region can define its own behavior for address handling, data reading, and writing.

#### **Design of the Memory Management Unit (MMU)**

The MMU is a central structure that manages multiple `Addressable` objects representing different memory regions. Its primary functions include:

- **Address Space Lookup**:  
   The `space` method iterates through all known memory regions to find the one containing a specified address.

- **Read Operations**:  
   The `Read` method uses the `space` function to locate the memory region containing the target address and reads data from it. If no matching region is found, it returns `0xff`, mimicking behaviors like the black bars shown during the Game Boy boot process when no cartridge is present.

- **Write Operations**:  
   The `Write` method finds the memory region corresponding to the target address and attempts to write data to it.

This design enables flexible and efficient memory management, allowing the emulator to handle complex memory access scenarios and laying the groundwork for future expansions, such as display control and audio.

<details><summary><b> Code </b></summary>

```go

// Addressable interface provides functions to read/write bytes in a given
// 16-bit address space.
type Addressable interface {
    // Contains returns true if the given address belongs to the address space.
    Contains(addr uint16) bool

    // Read returns the value stored at the given address.
    Read(addr uint16) uint8

    // Write attempts to store the given value at the given address if writable.
    Write(addr uint16, value uint8)
}

// MMU manages an arbitrary number of ordered address spaces. It also satisfies
// the Addressable interface.
type MMU struct {
    Spaces []Addressable
}

// Returns the first address space that can handle the requested address or nil.
func (m *MMU) space(addr uint16) Addressable {
    for _, space := range m.Spaces {
        if space.Contains(addr) {
            return space
        }
    }
    return nil
}

// Contains returns whether one of the address spaces known to the MMU contains
// the given address. The first address space in the internal list containing a
// given address will shadow any other that may contain it.
func (m *MMU) Contains(addr uint16) bool {
    return m.space(addr) != nil
}

// Read finds the first address space compatible with the given address and
// returns the value at that address. If no space contains the requested
// address, it returns 0xff (emulates black bar on boot).
func (m *MMU) Read(addr uint16) uint8 {
    if space := m.space(addr); space != nil {
        return space.Read(addr)
    }
    return 0xff
}

// Write finds the first address space compatible with the given address and
// attempts writing the given value to that address.
func (m *MMU) Write(addr uint16, value uint8) {
    if space := m.space(addr); space != nil {
        space.Write(addr, value)
    }
}

```

</details>

#### **Refactoring the Emulator: Integrating MMU for Modular Memory Management**

- **Replacing Memory Arrays with MMU**: Previously, memory was managed through direct manipulation of a byte array (e.g., `c.memory`). With the introduction of MMU, all memory accesses are now routed through `c.mmu.Read(<address>)` or `c.mmu.Write(<address>)`. The MMU decides whether a request targets RAM, Boot ROM, or another memory region. This restructuring not only improves code clarity but also simplifies future expansions.

- **Defining the RAM Type**: The **RAM type** represents a memory region with a start address and size. It implements the `Read`, `Write`, and `Contains` methods to manage data access and validate address ranges. This approach allows for flexible creation of memory blocks of varying sizes and addresses.

<details><summary><b> Code </b></summary>

```go

// RAM as an arbitrary long list of R/W bytes at addresses starting from a
// given offset.
type RAM struct {
    bytes []uint8
    start uint16
}

// NewRAM instantiates a zeroed slice of the given size to represent RAM.
func NewRAM(start, size uint16) *RAM {
    return &RAM{make([]uint8, size), start}
}

// Read returns the byte at the given address (adjusting for offset).
func (r *RAM) Read(addr uint16) uint8 {
    return r.bytes[addr-r.start]
}

// Write stores the given value at the given address (adjusting for offset).
func (r *RAM) Write(addr uint16, value uint8) {
    r.bytes[addr-r.start] = value
}

// Contains returns true as long as the given address fits in the slice.
func (r *RAM) Contains(addr uint16) bool {
    return addr >= r.start && addr < r.start+uint16(len(r.bytes))
}

```

</details>

- **Simulating Boot ROM**: A **Boot type** is defined to manage the Boot ROM's content and control register. By controlling the `0xff50` register, the emulator can enable or disable Boot ROM access. When disabled, the MMU automatically redirects memory requests to other regions.

<details><summary><b> Code </b></summary>

```go

// Boot address space translating memory access to Boot ROM and the BOOT
// register at address 0xff50.
type Boot struct {
    rom      RAM   // The contents of the boot ROM.
    register uint8 // BOOT register at address 0xff50.
    disabled bool  // Writing to 0xff50 will disable boot ROM access.
}

// NewBoot returns a new address space containing the boot ROM and the BOOT
// register.
func NewBoot(filename string) *Boot {
    rom, err := ioutil.ReadFile(filename)
    if err != nil {
        panic(err)
    }
    return &Boot{rom: RAM{rom, 0}}
}

// Contains returns true if the given address is that of the BOOT register,
// or if the boot ROM is not disabled and contains the address.
func (b *Boot) Contains(addr uint16) bool {
    return addr == 0xff50 || (!b.disabled && b.rom.Contains(addr))
}

// Read returns the value stored at the given address in ROM or BOOT register.
// If the boot ROM was disabled, Contains() should ensure this method will
// only be called with address 0xff50.
func (b *Boot) Read(addr uint16) uint8 {
    if addr == 0xff50 {
        return b.register
    }
    return b.rom.Read(addr)
}

// Write is only supported for the BOOT register and disables the boot ROM.
func (b *Boot) Write(addr uint16, value uint8) {
    if addr == 0xff50 {
        b.register = value
        b.disabled = true
    } // Writing to the boot ROM itself is obviously not allowed.
}

```

</details>

- **MMU as a Manager for Multiple Memory Regions**: With MMU, all memory access—whether for Boot ROM, RAM, or hardware registers—is centrally managed. The MMU dynamically routes requests based on the address. For example: Boot ROM access takes precedence until it is disabled. Once disabled, the MMU seamlessly redirects access to other memory regions.

The introduction of MMU divides memory management into discrete **Addressable** objects, each responsible for handling its respective memory region. Each region can have its unique logic for handling read, write, and address operations. This makes it easy to add new components, such as hardware registers or additional memory regions, without disrupting existing functionality.

---

## 1. SP1: A RISC-V Virtual Machine

**SP1** is a virtual machine designed to support the **RISC-V instruction set** while providing proof of correct execution for programs running on it. It features a comprehensive toolchain capable of compiling **Rust** code into **ELF** files (Executable and Linkable Format) for execution on the RISC-V architecture. SP1 employs the **LogUp** algorithm to describe chip interconnections, address memory access consistency challenges, and enhance circuit constraints for proof generation. The virtual machine is constructed around multiple components, including the CPU, arithmetic logic units (ALU), memory modules, and more. Additionally, SP1 extends the **Plonky3** proof system with enhanced AIR (Algebraic Intermediate Representation) for chip interaction and uses the **LogUp** algorithm to fully describe constraints for the RISC-V CPU.

#### **Core Design of SP1**

The architecture of SP1 is centered on **State** and **Actions** to facilitate efficient program execution and proof generation.

##### 1) **State Components**

- **ELF Program Loading**: Programs are loaded in ELF format, containing executable code and associated data.

- **Program Counter (PC)**: The PC tracks the address of the instruction being executed, serving as the core control mechanism for program flow.

- **Memory**: Memory modules store runtime data and instruction results. Registers are utilized for faster data operations.

- **Clock**: A clock keeps track of execution steps or loop iterations.

- **Witness Input**: Witness inputs, which remain private, influence program execution and are crucial for zero-knowledge proofs.

- **Public Values**: These are public outputs that the proof must validate, typically representing program results.

##### 2) **Actions in SP1**

- **Instruction Fetch and Decode**: The virtual machine fetches the instruction pointed to by the PC from memory and decodes it to determine its type and operations.

- **Instruction Execution**: Based on the decoded result, the appropriate logic unit is invoked to execute the instruction, updating registers or memory states as required.

- **Program Termination Check**: Execution terminates when the PC value equals `0`, signaling program completion.

##### 3) **Execution Flow**

The main execution loop in SP1 repeatedly fetches, decodes, and executes instructions until the program exits. Instructions are classified into categories like arithmetic, logic, memory operations, etc. Each instruction modifies the machine state, including updates to memory, registers, the PC, or public and witness values.

#### **Proof Generation with Plonky3**

SP1 leverages the **Plonky3** framework to generate zero-knowledge proofs. Plonky3 employs AIR to describe circuit constraints and uses polynomial commitment schemes for proof generation.

- **Core Prove Function**  
   The `prove` function builds proofs using configuration parameters, the challenger, circuit constraints (AIR), trace data (execution traces), and public values.

- **Constraint Builders**
  - **DebugConstraintBuilder**: Ensures constraints are satisfied at every row.
  - **ProverConstraintFolder**: Constructs quotient polynomials for constraints.
  - **SymbolicAirBuilder**: Determines the degree of quotient polynomials.

<details><summary><b> Code </b></summary>

```rust

pub fn prove<
    SC,
    #[cfg(debug_assertions)] A: for<'a> Air<crate::check_constraints::DebugConstraintBuilder<'a, Val<SC>>>,
    #[cfg(not(debug_assertions))] A,
>(
    config: &SC,
    air: &A,
    challenger: &mut SC::Challenger,
    trace: RowMajorMatrix<Val<SC>>,
    public_values: &Vec<Val<SC>>,
) -> Proof<SC>
where
    SC: StarkGenericConfig,
    A: Air<SymbolicAirBuilder<Val<SC>>> + for<'a> Air<ProverConstraintFolder<'a, SC>>,
{

...

pub trait Air<AB: AirBuilder>: BaseAir<AB::F> {
    fn eval(&self, builder: &mut AB);
}

```

</details>

Plonky3 transforms circuit constraints into polynomial form and employs polynomial commitments and openings to produce proofs.

#### **SP1 Toolchain**

SP1 provides a seamless workflow from program compilation to proof generation via the `cargo prove` command. The toolchain includes the following:

- **new**: Creates a project template with program and script directories.
- **build**: Compiles the program into an ELF file.
- **prove**: The default command for compilation and proof generation.
- **build-toolchain**: Builds the `cargo-prove` toolchain.
- **install-toolchain**: Installs the toolchain.
- **help**: Displays help information.

<details><summary><b> Code </b></summary>

```rust


let result = Command::new("cargo")
            .env("RUSTUP_TOOLCHAIN", "succinct")
            .env("CARGO_ENCODED_RUSTFLAGS", rust_flags.join("\x1f"))
            .args(&cargo_args)
            .status()
            .context("Failed to run cargo command.")?;

let client = ProverClient::new();
let (pk, _) = client.setup(&elf);
let proof = client.prove(&pk, stdin).unwrap();

```

</details>

The proof workflow includes the steps of generating, verifying, and optimizing zero-knowledge proofs to ensure their correctness and efficiency:

- **Compiling Programs**  
   Rust code is compiled into ELF files using the RISC-V instruction set (`riscv32im`). Configuration includes flags for single-threaded execution and specifying the program’s starting address.

- **Generating Proofs**  
   The `prove` subcommand, implemented through `ProverClient`, orchestrates the proof process. Key steps include setting up the proving environment, generating proofs, and verifying them.

- **Prover Types**  
   SP1 offers three prover implementations: `MockProver`, `LocalProver`, and `NetworkProver`. The primary focus is on `LocalProver`, which encapsulates core logic for setup, proof generation, and verification.

#### **StarkMachine: A RISC-V-Based Virtual Machine**

SP1’s virtual machine, **StarkMachine**, is modeled using AIR and comprises multiple chips, each responsible for distinct functions:

- **CpuChip**  
   Manages program execution order, registers, and instruction validity.

- **MemoryChip**  
   Handles memory operations, including initialization and final state verification.

- **ProgramChip**  
   Stores program code and tracks execution details such as the program counter (PC).

- **MemoryProgramChip**  
   Manages fixed memory allocations for the program.

- **ByteChip**  
   Processes byte-level operations like logic and range checks.

- **ALU Chip**  
   Executes arithmetic operations (e.g., addition, subtraction).

<details><summary><b> Code </b></summary>

```rust

pub enum MemoryChipType {
    Initialize,
    Finalize,
}

pub struct MemoryInitCols<T> {
    pub shard: T,
    pub timestamp: T,
    pub addr: T,
    pub value: Word<T>,
    pub is_real: T,
}

pub struct ProgramPreprocessedCols<T> {
    pub pc: T,
    pub instruction: InstructionCols<T>,
    pub selectors: OpcodeSelectorCols<T>,
}
pub struct ProgramMultiplicityCols<T> {
    pub shard: T,
    pub multiplicity: T,
}

pub struct MemoryProgramPreprocessedCols<T> {
    pub addr: T,
    pub value: Word<T>,
    pub is_real: T,
}
pub struct MemoryProgramMultCols<T> {
    pub multiplicity: T,
    pub is_first_shard: IsZeroOperation<T>,
}

pub struct BytePreprocessedCols<T> {
    pub b: T,
    pub c: T,
    pub and: T,
    pub or: T,
    pub xor: T,
    pub sll: T,
    pub shr: T,
    pub shr_carry: T,
    pub ltu: T,
    pub msb: T,
    pub value_u16: T,
}

pub struct MultiplicitiesCols<T> {
    pub multiplicities: [T; NUM_BYTE_OPS],
}

pub struct ByteMultCols<T> {
    pub shard: T,
    pub mult_channels: [MultiplicitiesCols<T>; NUM_BYTE_LOOKUP_CHANNELS as usize],
}

pub struct AddSubCols<T> {
    pub shard: T,
    pub channel: T,
    pub add_operation: AddOperation<T>,
    pub operand_1: Word<T>,
    pub operand_2: Word<T>,
    pub is_add: T,
    pub is_sub: T,
}

pub struct Chip<F: Field, A> {
    /// The underlying AIR of the chip for constraint evaluation.
    air: A,
    /// The interactions that the chip sends.
    sends: Vec<Interaction<F>>,
    /// The interactions that the chip receives.
    receives: Vec<Interaction<F>>,
    /// The relative log degree of the quotient polynomial, i.e. `log2(max_constraint_degree - 1)`.
    log_quotient_degree: usize,
}

```

</details>

#### **Chip Interactions**

Interactions between chips are defined as “send” and “receive” operations. These are managed through the `Interaction` structure and support tasks like memory operations and byte-level processing.

<details><summary><b> Code </b></summary>

```rust


pub struct Interaction<F: Field> {
    pub values: Vec<VirtualPairCol<F>>,
    pub multiplicity: VirtualPairCol<F>,
    pub kind: InteractionKind,
}

pub enum InteractionKind {
    /// Interaction with the memory table, such as read and write.
    Memory = 1,
    /// Interaction with the program table, loading an instruction at a given pc address.
    Program = 2,
    /// Interaction with instruction oracle.
    Instruction = 3,
    /// Interaction with the ALU operations.
    Alu = 4,
    /// Interaction with the byte lookup table for byte operations.
    Byte = 5,
    /// Requesting a range check for a given value and range.
    Range = 6,
    /// Interaction with the field op table for field operations.
    Field = 7,
    /// Interaction with a syscall.
    Syscall = 8,
}

```

</details>

#### **Memory Consistency**

SP1 ensures memory consistency using the **LogUp** algorithm. Memory read and write operations are transformed into permutation checks, ensuring data correctness.

<details><summary><b> Code </b></summary>

```rust

fn eval_memory_access<E: Into<Self::Expr> + Clone>(
    &mut self,
    shard: impl Into<Self::Expr>,
    channel: impl Into<Self::Expr>,
    clk: impl Into<Self::Expr>,
    addr: impl Into<Self::Expr>,
    memory_access: &impl MemoryCols<E>,
    do_check: impl Into<Self::Expr>,
) {
...
        // The previous values get sent with multiplicity = 1, for "read".
        self.send(AirInteraction::new(
            prev_values,
            do_check.clone(),
            InteractionKind::Memory,
        ));

        // The current values get "received", i.e. multiplicity = -1
        self.receive(AirInteraction::new(
            current_values,
            do_check.clone(),
            InteractionKind::Memory,
        ));
}

```

</details>

#### The execution process of the SP1 virtual machine

- **Execution Framework**: Program execution takes place within a **Runtime environment** that records and manages various events. The Runtime environment is primarily composed of three parts: `IO Management` handles input and output operations, `ExecutionState` stores intermediate states during program execution and `ExecutionRecord` logs the events occurring during execution. Programs are divided into multiple **Shards**, with each Shard containing a set number of instructions. Execution is controlled by the `execute` function, while the execution of individual Shards is managed through the `execute_cycle` function.

<details><summary><b> Code </b></summary>

```rust


fn execute(&mut self) -> Result<bool, ExecutionError> {
    if self.state.global_clk == 0 {
        self.initialize();
    }
    ...
    loop {
        if self.execute_cycle()? {
            done = true;
            break;
        }
        ...
    }

    if done {
        self.postprocess();
    }

    Ok(done)
}

pub struct SP1CoreOpts {
    pub shard_size: usize,
    pub shard_batch_size: usize,
    pub shard_chunking_multiplier: usize,
    pub reconstruct_commitments: bool,
}

impl Default for SP1CoreOpts {
    fn default() -> Self {
        Self {
            shard_size: 1 << 22,
            shard_batch_size: 16,
            shard_chunking_multiplier: 1,
            reconstruct_commitments: false,
        }
    }
}

```

</details>

- **Shards and Tracing**: Each batch of instructions (typically $2^{22}$ instructions) constitutes a **Shard**. For each Shard, corresponding event data is organized, and trace data is generated for every chip involved in the execution.

- **Proof Generation**: Using the trace data and **Algebraic Intermediate Representation (AIR)** constraints, the `prove_shard` function generates proofs for each Shard. These proofs validate the connections and logical consistency across chips. The proof generation process utilizes algorithms akin to **Plonky3**, involving techniques such as polynomial representations and cryptographic commitments.

<details><summary><b> Code </b></summary>

```rust

pub fn prove_shard(
    config: &SC,
    pk: &StarkProvingKey<SC>,
    chips: &[&MachineChip<SC, A>],
    mut shard_data: ShardMainData<SC>,
    challenger: &mut SC::Challenger,
) -> ShardProof<SC>

```

</details>

- **Recursive Proofs**: To minimize verification overhead, SP1 supports **recursive proof aggregation**, where proofs from multiple Shards are combined into a single proof. This aggregated proof can then be verified on-chain using the **Groth16** algorithm, significantly reducing computational complexity for the verifier.

---

## 2. RISC-V: An Open-source Reduced ISA

**RISC-V** is an open-source reduced instruction set architecture (RISC) that features a modular design, allowing customization of the instruction set based on specific needs. Compared to complex instruction set computers (CISC), RISC architectures optimize the instruction set with fewer and more unified instructions. Some of the core features of RISC-V include:

- **Load/Store Architecture**: Memory access is performed via specific load and store instructions, rather than allowing most instructions to directly manipulate memory.
- **Single-Cycle Instructions**: Most instructions are completed in a single clock cycle.
- **Efficient Pipelining**: The instruction execution order is optimized, reducing the cycles per instruction (CPI).
- **Core Components**:
  - **CPU**: Responsible for executing instructions, including registers, the program counter (PC), and the arithmetic logic unit (ALU).
  - **DRAM**: Stores instructions and data.
  - **Bus**: Connects the CPU, memory, and peripheral devices, transferring data.
- **Basic Execution Flow**: Instructions are fetched from memory, executed based on their type (load, store, arithmetic, or branch), and the program counter (PC) is updated to point to the next instruction.

### 2.1 **Simple RISC-V Simulator (Rust Version)**

This simplified RISC-V instruction set simulator implements the RV32IM instruction set (also known as **rrs**), which includes basic operations such as arithmetic, logic, branching, load/store, and multiplication/division, but does not handle operating system-level functionality like exception handling and memory management. It also omits more complex extensions like atomic operations. This simulator is used for emulating the RISC-V instruction set and can be utilized for CPU performance modeling and verification.

To maintain modularity, the simulator is divided into two parts:

- **rrs-lib**: The core simulation library. It is a flexible component collection designed to build instruction set simulators (ISS). Users can utilize **rrs-lib** to create the specific functionality they need without relying on a large monolithic simulator. For instance, users can use **rrs-lib** to simulate the RISC-V instruction set and build performance models for CPU cores or perform co-simulation during CPU design verification.
- **rrs-cli**: A command-line tool for executing RISC-V programs using the **rrs-lib**. It allows users to set up simulated memory, load and execute compiled RISC-V binary files, and provides standard output devices for simulation termination.

#### **Instruction Execution**

When building an instruction set simulator, one common approach is using `switch-case` statements to handle different instructions. However, as the number of instructions increases, the efficiency of such a method can decrease. To improve performance, a jump table can be used, where the opcode directly maps to the corresponding execution function. In Rust, a similar approach is achieved using the `match` statement, which offers powerful pattern matching capabilities.

To ensure modularity and extensibility, the simulator leverages Rust’s `trait` mechanism. A `trait` is similar to interfaces in other languages and defines functions for handling different RISC-V instructions. For example, a `InstructionProcessor` trait may include functions for processing different RISC-V instructions. Each function accepts a decoded instruction and returns the execution result.

<details><summary><b> Code </b></summary>

```rust


pub trait InstructionProcessor {
    type InstructionResult;

    fn process_add(&mut self, dec_insn: instruction_formats::RType) -> Self::InstructionResult;
    fn process_or(&mut self, dec_insn: instruction_formats::RType) -> Self::InstructionResult;

    // ...

    fn process_addi(&mut self, dec_insn: instruction_formats::IType) -> Self::InstructionResult;
    fn process_ori(&mut self, dec_insn: instruction_formats::IType) -> Self::InstructionResult;

    // ...

    fn process_lui(&mut self, dec_insn: instruction_formats::UType) -> Self::InstructionResult;
    fn process_auipc(&mut self, dec_insn: instruction_formats::UType) -> Self::InstructionResult;

    // ...

    fn process_beq(&mut self, dec_insn: instruction_formats::BType) -> Self::InstructionResult;
    fn process_bne(&mut self, dec_insn: instruction_formats::BType) -> Self::InstructionResult;
    fn process_blt(&mut self, dec_insn: instruction_formats::BType) -> Self::InstructionResult;

    // ...

    fn process_lw(&mut self, dec_insn: instruction_formats::IType) -> Self::InstructionResult;
    fn process_sw(&mut self, dec_insn: instruction_formats::SType) -> Self::InstructionResult;

    // ...

    fn process_jal(&mut self, dec_insn: instruction_formats::JType) -> Self::InstructionResult;

    // ...

    fn process_mul(&mut self, dec_insn: instruction_formats::RType) -> Self::InstructionResult;
    fn process_div(&mut self, dec_insn: instruction_formats::RType) -> Self::InstructionResult;

    // ...
}

```

</details>

Rust’s struct types represent different instruction formats (e.g., R-Type, I-Type), each with a constructor to instantiate from encoded instructions. This design modularizes instruction processing, allowing for easy extension of the logic to support various instruction types. The `process_instruction` function processes each decoded instruction using an appropriate `InstructionProcessor` and invokes the relevant function, such as `process_opcode_op` for ALU-type instructions. This structure behaves like a `match` statement, using the opcode to select the appropriate handler.

<details><summary><b> Code </b></summary>

```rust


pub struct RType {
    pub funct7: u32,
    pub rs2: usize,
    pub rs1: usize,
    pub funct3: u32,
    pub rd: usize,
}

impl RType {
    pub fn new(insn: u32) -> RType {
        RType {
            funct7: (insn >> 25) & 0x7f,
            rs2: ((insn >> 20) & 0x1f) as usize,
            rs1: ((insn >> 15) & 0x1f) as usize,
            funct3: (insn >> 12) & 0x7,
            rd: ((insn >> 7) & 0x1f) as usize,
        }
    }
}

pub fn process_instruction<T: InstructionProcessor>(
    processor: &mut T,
    insn_bits: u32,
) -> Option<T::InstructionResult> {
    let opcode: u32 = insn_bits & 0x7f;

    match opcode {
        instruction_formats::OPCODE_OP => process_opcode_op(processor, insn_bits),
        instruction_formats::OPCODE_OP_IMM => process_opcode_op_imm(processor, insn_bits),
        instruction_formats::OPCODE_LUI => {
            Some(processor.process_lui(instruction_formats::UType::new(insn_bits)))
        }
        instruction_formats::OPCODE_AUIPC => {
            Some(processor.process_auipc(instruction_formats::UType::new(insn_bits)))
        }
        instruction_formats::OPCODE_BRANCH => process_opcode_branch(processor, insn_bits),
        instruction_formats::OPCODE_LOAD => process_opcode_load(processor, insn_bits),
        instruction_formats::OPCODE_STORE => process_opcode_store(processor, insn_bits),
        instruction_formats::OPCODE_JAL => {
            Some(processor.process_jal(instruction_formats::JType::new(insn_bits)))
        }
        instruction_formats::OPCODE_JALR => {
            Some(processor.process_jalr(instruction_formats::IType::new(insn_bits)))
        }
        _ => None,
    }
}

```

</details>

Using Rust's generics eliminates the performance overhead of virtual function calls, enabling better compiler optimizations such as function inlining, resulting in faster execution. This design not only modularizes instruction processing but also ensures high performance, making it easy to extend and optimize.

#### **CPU State (`HartState`)**

The **CPU State** is represented by the `HartState` struct, which represents a hardware thread (`hart`). It includes:

- **Registers**: Stores the values of 32 general-purpose registers (x0 to x31). The value of register `x0` is always 0, representing the zero register.
- **Program Counter (PC)**: Keeps track of the current instruction address.
- **Last Register Write**: Tracks the index of the last register written to, or `None` if no register was written.

The struct offers two functions:

- `write_register`: Writes a value to a register, skipping `x0`.
- `read_register`: Reads the value of a register, returning 0 when `x0` is read.

<details><summary><b> Code </b></summary>

```rust

pub struct HartState {
    /// x1 - x31 register values. The contents of index 0 (the x0 zero register) are ignored.
    pub registers: [u32; 32],
    /// Program counter
    pub pc: u32,
    /// Gives index of the last register written if one occurred in the previous instruciton. Set
    /// to `None` if latest instruction did not write a register.
    pub last_register_write: Option<usize>,
}

impl HartState {
    /// Write a register in the hart state. Used by executing instructions for correct zero
    /// register handling
    fn write_register(&mut self, reg_index: usize, data: u32) {
        if reg_index == 0 {
            return;
        }

        self.registers[reg_index] = data;
        self.last_register_write = Some(reg_index)
    }

    /// Read a register from the hart state. Used by executing instructions for correct zero
    /// register handling
    fn read_register(&self, reg_index: usize) -> u32 {
        if reg_index == 0 {
            0
        } else {
            self.registers[reg_index]
        }
    }
}


pub struct InstructionExecutor<'a, M: Memory> {
    /// Memory used by load and store instructions
    pub mem: &'a mut M,
    pub hart_state: &'a mut HartState,
}

```

</details>

#### **Instruction Executor (`InstructionExecutor`)**

The `InstructionExecutor` struct is responsible for executing instructions. It contains:

- **Memory**: Used for loading and storing data.
- **HartState**: A reference to the current CPU state, representing the state of the `hart`.

This struct implements the `InstructionProcessor` trait and handles actual instruction execution. Each instruction execution returns an `InstructionResult`, which is a `Result<bool, InstructionException>`. If the execution is successful, `Ok(false)` is returned, indicating that the instruction did not modify the program counter. If an exception occurs, an error is returned with the exception details.

#### **Reducing Code Duplication**

Many instructions, such as addition and subtraction, have similar structures with the main difference being the operation. To reduce code duplication, the **`execute_reg_reg_op` function** is used to handle register-to-register operations by accepting an operation closure. This reduces redundant code for instructions like `add` and `sub`.

<details><summary><b> Code </b></summary>

```rust

fn execute_reg_reg_op<F>(&mut self, dec_insn: instruction_formats::RType, op: F)
where
    F: Fn(u32, u32) -> u32,
{
    let a = self.hart_state.read_register(dec_insn.rs1);
    let b = self.hart_state.read_register(dec_insn.rs2);
    let result = op(a, b);
    self.hart_state.write_register(dec_insn.rd, result);
}

```

</details>

Rust macros are also employed to generate repetitive code. For example, the **`make_alu_op_reg_fn!`** macro generates operation functions, avoiding the need to manually write similar code. To further reduce repetition for instructions like `add` and `addi`, the **`make_alu_op_fns!`** macro generates handlers for both register-to-register and register-to-immediate operations.

<details><summary><b> Code </b></summary>

```rust


macro_rules! make_alu_op_reg_fn {
    ($name:ident, $op_fn:expr) => {
        paste! {
            fn [<process_ $name>](
                &mut self,
                dec_insn: instruction_formats::RType
            ) -> Self::InstructionResult {
                self.execute_reg_reg_op(dec_insn, $op_fn);

                Ok(false)
            }
        }
    };
}

macro_rules! make_alu_op_fns {
    ($name:ident, $op_fn:expr) => {
        make_alu_op_reg_fn! {$name, $op_fn}
        make_alu_op_imm_fn! {$name, $op_fn}
    };
}

make_alu_op_fns! {add, |a, b| a.wrapping_add(b)}
make_alu_op_reg_fn! {sub, |a, b| a.wrapping_sub(b)}
make_alu_op_fns! {slt, |a, b| if (a as i32) < (b as i32) {1} else {0}}

```

</details>

#### **Memory Management**

The simulator implements flexible memory management through the `Memory` trait and `MemorySpace` struct, supporting various memory types such as regular RAM and device memory. Rust’s ownership system and the `Downcast` trait ensure safe memory operations and type-safe conversion.

The **`Memory` Trait** defines two core operations:

- `read_mem`: Reads data of a specified size from memory.
- `write_mem`: Writes data of a specified size to memory.

<details><summary><b> Code </b></summary>

```rust


pub trait Memory: Downcast {
    fn read_mem(&mut self, addr: u32, size: MemAccessSize) -> Option<u32>;
    fn write_mem(&mut self, addr: u32, size: MemAccessSize, store_data: u32) -> bool;
}

```

</details>

The **`MemorySpace` Struct** manages multiple memory regions, such as RAM and device memory. It allows different types of memory to be added to the space via the `add_memory` function. The `add_memory` function accepts a memory address, size, and a memory object that implements the `Memory` trait, returning an index for referencing the memory.

<details><summary><b> Code </b></summary>

```rust

pub fn add_memory(
    &mut self,
    base: u32,
    size: u32,
    memory: Box<dyn Memory>,
) -> Result<usize, MemorySpaceError> {
    //...
}

```

</details>

Rust’s memory management system guarantees safety by enforcing ownership and borrowing rules. In `MemorySpace`, memory ownership is transferred, ensuring that no references are left behind. To address this, `MemorySpace` provides methods to obtain memory references and uses `Downcast` to safely convert between types.

#### **Running the Simulator**

The `InstructionExecutor::step` function is responsible for fetching and executing instructions. The process is as follows:

- **Fetch Instruction**: The instruction located at the memory address indicated by the program counter (PC) is fetched using the `read_mem` function.
- **Process Instruction**: The fetched instruction is passed to the `process_instruction` function, which handles its execution and returns an `Option<Result<bool, InstructionException>>`. The result indicates:
  - **Failure**: If instruction decoding or execution fails, an exception is returned.
  - **Success**: If the instruction executes successfully, the result determines whether the PC has been explicitly updated. If not, the PC is incremented by 4 (the standard increment for 32-bit RISC-V instructions).

<details><summary><b> Code </b></summary>

```rust


/// Execute instruction pointed to by `hart_state.pc`
///
/// Returns `Ok` where instruction execution was successful. `Err` with the relevant
/// [InstructionException] is returned when the instruction execution causes an exception.
pub fn step(&mut self) -> Result<(), InstructionException> {
    self.hart_state.last_register_write = None;

    // Fetch next instruction from memory
    if let Some(next_insn) = self.mem.read_mem(self.hart_state.pc, MemAccessSize::Word) {
        // Execute the instruction
        let step_result = process_instruction(self, next_insn);

        match step_result {
            Some(Ok(pc_updated)) => {
                if !pc_updated {
                    // Instruction didn't update PC so increment to next instruction
                    self.hart_state.pc += 4;
                }
                Ok(())
            }
            // Instruction produced an error so return it
            Some(Err(e)) => Err(e),
            // Instruction decode failed so return an IllegalInstruction as an error
            None => Err(InstructionException::IllegalInstruction(
                self.hart_state.pc,
                next_insn,
            )),
        }
    } else {
        // Return a FetchError as an error if instruction fetch fails
        Err(InstructionException::FetchError(self.hart_state.pc))
    }
}

```

</details>

The **`run_sim`** function is responsible for running the simulation. It initializes an `InstructionExecutor`, which references the `HartState` (CPU state) and `MemorySpace` (memory space). During each instruction cycle, the `step` function is called to fetch and execute instructions.

The `step` function reads the instruction from memory at the address specified by the PC, processes it, and returns a result. If an exception occurs, it is handled; otherwise, the program counter is updated. If the instruction does not modify the PC, it defaults to incrementing by 4.

During simulation, the `run_sim` function logs the disassembly of instructions and updates the registers (if configured). It checks for termination requests through the `SimulationCtrlDevice`. Once the simulation ends, statistics like the total number of instructions executed, the execution time, and simulation frequency (in MHz) are computed.

<details><summary><b> Code </b></summary>

```rust


fn run_sim(sim_environment: &mut SimEnvironment) {
    let mut executor = InstructionExecutor {
        hart_state: &mut sim_environment.hart_state,
        mem: &mut sim_environment.memory_space,
    };

    let mut insn_count: u64 = 0;
    let start = Instant::now();

    loop {
        if let Some(log_file) = &mut sim_environment.log_file {
            // Output current instruction disassembly to log
            let insn_bits = executor
                .mem
                .read_mem(executor.hart_state.pc, MemAccessSize::Word)
                .unwrap_or_else(|| panic!("Could not read PC {:08x}", executor.hart_state.pc));

            let mut outputter = InstructionStringOutputter {
                insn_pc: executor.hart_state.pc,
            };

            writeln!(
                log_file,
                "{:x} {}",
                executor.hart_state.pc,
                rrs_lib::process_instruction(&mut outputter, insn_bits).unwrap()
            ).expect("Log file write failed");
        }

        // Execute instruction
        executor.step().expect("Exception during execution");

        insn_count += 1;

        // Stop if stop requested by emulated binary via SimulationCtrlDevice
        if executor
            .mem
            .get_memory_ref::<SimulationCtrlDevice>(sim_environment.sim_ctrl_dev_idx)
            .unwrap()
            .stop
        {
            break;
        }

        if let Some(log_file) = &mut sim_environment.log_file {
            if let Some(reg_index) = executor.hart_state.last_register_write {
                // Output register written by instruction to log if it wrote to one
                writeln!(
                    log_file,
                    "x{} = {:08x}",
                    reg_index, executor.hart_state.registers[reg_index]
                ).expect("Log file write failed");
            }
        }
    }

    let elapsed = start.elapsed();
    let mhz = (insn_count as f64) / (elapsed.as_micros() as f64);
    println!(
        "{} instructions executed in {} ms {} MHz",
        insn_count,
        elapsed.as_millis(),
        mhz
    );
}

```

</details>

---

### 2.2 Simple RISC-V Simulator (C Version)

**DRAM (Dynamic Random Access Memory):**  
DRAM serves as the core memory to store instructions and data in the RISC-V simulator. It is implemented as an array holding 64-bit values, with its size defined by `DRAM_SIZE` and the starting address specified by `DRAM_BASE`.

RISC-V architecture employs **memory-mapped I/O**, where memory address space is shared between memory and I/O devices. For instance, in QEMU, low addresses are reserved for I/O ports, while DRAM begins at `0x80000000`. This approach enables the CPU to access I/O devices using standard memory operation instructions.

**DRAM Structure:**  
A memory structure represents DRAM. The CPU interacts with DRAM via two primary functions:

- **`dram_load()`**: Reads data from a specified address in DRAM.
- **`dram_store()`**: Writes data to a specified address in DRAM.

These functions operate based on the size of the data (8, 16, 32, or 64 bits), invoking private helper functions for specific sizes.  
The main `dram_load()` and `dram_store()` functions use a `switch` statement to determine which helper function to call.

<details><summary><b> Code </b></summary>

To represent DRAM, we define a memory structure as follows:

```c


// includes/dram.h
#define DRAM_SIZE 1024*1024*1     // 1 MiB DRAM
#define DRAM_BASE 0x80000000      // Starting address of DRAM

typedef struct DRAM {
    uint8_t mem[DRAM_SIZE];       // Memory array representing DRAM
} DRAM;

```

The CPU interacts with DRAM to read instructions and data, as well as to write data. Two key functions are defined:

- **`dram_load()`**: Reads data from a specified address.
- **`dram_store()`**: Writes data to a specified address.

```c

// includes/dram.h
uint64_t dram_load(DRAM* dram, uint64_t addr, uint64_t size);
void dram_store(DRAM* dram, uint64_t addr, uint64_t size, uint64_t value);

```

The implementation includes reading and writing data of different sizes (8, 16, 32, 64 bits). Private functions handle the logic for these operations.

```c

// src/dram.c
// For a 32-bit read (`dram_load_32`), the memory address is adjusted by subtracting `DRAM_BASE`. Using little-endian format, the lower byte is stored at the lower address. The data is constructed by left-shifting higher bytes into the correct positions.

uint64_t dram_load_32(DRAM* dram, uint64_t addr){
    return (uint64_t) dram->mem[addr-DRAM_BASE]
        |  (uint64_t) dram->mem[addr-DRAM_BASE + 1] << 8
        |  (uint64_t) dram->mem[addr-DRAM_BASE + 2] << 16
        |  (uint64_t) dram->mem[addr-DRAM_BASE + 3] << 24;
}

```

```c


// Similarly, for a 64-bit read (`dram_load_64`), 8 bytes are processed:

uint64_t dram_load_64(DRAM* dram, uint64_t addr){
    return (uint64_t) dram->mem[addr-DRAM_BASE]
        |  (uint64_t) dram->mem[addr-DRAM_BASE + 1] << 8
        |  (uint64_t) dram->mem[addr-DRAM_BASE + 2] << 16
        |  (uint64_t) dram->mem[addr-DRAM_BASE + 3] << 24
        |  (uint64_t) dram->mem[addr-DRAM_BASE + 4] << 32
        |  (uint64_t) dram->mem[addr-DRAM_BASE + 5] << 40
        |  (uint64_t) dram->mem[addr-DRAM_BASE + 6] << 48
        |  (uint64_t) dram->mem[addr-DRAM_BASE + 7] << 56;
}
```

Writing data follows a similar approach. For a 16-bit write (`dram_store_16`), the lower byte is written to the lower address, and the higher byte is written to the higher address.

```c


void dram_store_16(DRAM* dram, uint64_t addr, uint64_t value) {
    dram->mem[addr-DRAM_BASE] = (uint8_t) (value & 0xff);
    dram->mem[addr-DRAM_BASE + 1] = (uint8_t) ((value >> 8) & 0xff);
}

```

For a 64-bit write (`dram_store_64`), all 8 bytes are processed:

```c


void dram_store_64(DRAM* dram, uint64_t addr, uint64_t value) {
    dram->mem[addr-DRAM_BASE] = (uint8_t) (value & 0xff);
    dram->mem[addr-DRAM_BASE + 1] = (uint8_t) ((value >> 8) & 0xff);
    dram->mem[addr-DRAM_BASE + 2] = (uint8_t) ((value >> 16) & 0xff);
    dram->mem[addr-DRAM_BASE + 3] = (uint8_t) ((value >> 24) & 0xff);
    dram->mem[addr-DRAM_BASE + 4] = (uint8_t) ((value >> 32) & 0xff);
    dram->mem[addr-DRAM_BASE + 5] = (uint8_t) ((value >> 40) & 0xff);
    dram->mem[addr-DRAM_BASE + 6] = (uint8_t) ((value >> 48) & 0xff);
    dram->mem[addr-DRAM_BASE + 7] = (uint8_t) ((value >> 56) & 0xff);
}

```

The main functions `dram_load()` and `dram_store()` utilize a `switch` statement to handle different data sizes by invoking the corresponding functions.

```c

uint64_t dram_load(DRAM* dram, uint64_t addr, uint64_t size) {
    switch (size) {
        case 8:  return dram_load_8(dram, addr);  break;
        case 16: return dram_load_16(dram, addr); break;
        case 32: return dram_load_32(dram, addr); break;
        case 64: return dram_load_64(dram, addr); break;
        default: ;
    }
    return 1;  // Return 1 for invalid size
}

void dram_store(DRAM* dram, uint64_t addr, uint64_t size, uint64_t value) {
    switch (size) {
        case 8:  dram_store_8(dram, addr, value);  break;
        case 16: dram_store_16(dram, addr, value); break;
        case 32: dram_store_32(dram, addr, value); break;
        case 64: dram_store_64(dram, addr, value); break;
        default: ;
    }
}

```

</details>

#### **BUS (Bus Interface)**

The **BUS** facilitates data transfer between the CPU and DRAM. It includes an **address bus** and a **data bus**, with a 64-bit width for the 64-bit RISC-V implementation.  
In the simulator, the BUS connects to DRAM and provides unified access for loading and storing data. By abstracting the interaction through the BUS, the architecture becomes modular and clear.

The `BUS` structure contains a DRAM object representing the memory device connected to the BUS. Two functions handle data transfers:

- **`bus_load()`**: Loads data from a specified address via the BUS.
- **`bus_store()`**: Stores data to a specified address via the BUS.

Internally, these functions delegate the operations to the DRAM module’s `dram_load()` and `dram_store()` functions.

<details><summary><b> Code </b></summary>

```c

// includes/bus.h

typedef struct BUS {
    struct DRAM dram;  // The DRAM object connected to the BUS
} BUS;

uint64_t bus_load(BUS* bus, uint64_t addr, uint64_t size);
void bus_store(BUS* bus, uint64_t addr, uint64_t size, uint64_t value);

```

</details>

This definition ensures that all memory-related operations conducted through the bus will ultimately interact with the DRAM object.

The `bus_load()` function handles data loading from a specified address via the bus. Under the hood, it simply calls the `dram_load()` function of the DRAM module. This allows the bus to abstract the memory access mechanism, making the CPU-to-memory interaction modular and straightforward.

Similarly, the `bus_store()` function handles data storage at a specified address via the bus. This function calls `dram_store()` from the DRAM module, delegating the actual memory operation to the connected DRAM object.

<details><summary><b> Code </b></summary>

```c

// bus.c

uint64_t bus_load(BUS* bus, uint64_t addr, uint64_t size) {
    return dram_load(&(bus->dram), addr, size);  // Access DRAM through the bus
}

void bus_store(BUS* bus, uint64_t addr, uint64_t size, uint64_t value) {
    dram_store(&(bus->dram), addr, size, value);  // Write to DRAM through the bus
}

```

</details>

#### **RISC-V CPU Structure and Functions**

The RISC-V CPU simulator uses a structure containing three primary components:

1. **Registers**: 32 general-purpose 64-bit registers (`x0` is fixed at 0, while others store values).
2. **Program Counter (PC)**: A special-purpose register storing the address of the current instruction.
3. **BUS**: The CPU interacts with memory (DRAM) via the BUS.

<details><summary><b> Code </b></summary>

```c

// includes/cpu.h

#include <stdint.h>

typedef struct CPU {
    uint64_t regs[32];          // 32 64-bit general-purpose registers (x0-x31)
    uint64_t pc;                // 64-bit program counter
    struct BUS bus;             // Connected BUS object
} CPU;

void cpu_init(struct CPU *cpu);            // Initialize the CPU
uint32_t cpu_fetch(struct CPU *cpu);       // Fetch the instruction
int cpu_execute(struct CPU *cpu, uint32_t inst); // Decode and execute the instruction
void dump_registers(struct CPU *cpu);      // Debugging: output register states

```

</details>

##### 1) **CPU Initialization (`cpu_init`):**

- All registers are initialized to 0.
- The program counter (`PC`) is set to the start address of DRAM (`DRAM_BASE`).
- The stack pointer (`x2`) is initialized to the top of DRAM (`DRAM_BASE + DRAM_SIZE`).

<details><summary><b> Code </b></summary>

```c

// cpu.c

void cpu_init(CPU *cpu) {
    cpu->regs[0] = 0x00;                    // x0 is hardwired to 0
    cpu->regs[2] = DRAM_BASE + DRAM_SIZE;   // Set stack pointer (x2)
    cpu->pc      = DRAM_BASE;               // Set program counter to memory base
}

```

</details>

##### 2) **Instruction Fetch (`cpu_fetch`):**

The CPU reads the instruction from the memory address pointed to by the `PC` using the BUS interface. The fetched instruction is stored locally in the simulator.

<details><summary><b> Code </b></summary>

```c

// cpu.c

uint32_t cpu_fetch(struct CPU *cpu) {
    return bus_load(&(cpu->bus), cpu->pc, 32);  // Load 32-bit instruction from memory
}

```

</details>

##### 3) **Data Load and Store Functions (`cpu_load` and `cpu_store`):**

These functions provide an interface for the CPU to load data from or store data into memory via the BUS. Internally, they call the `bus_load()` and `bus_store()` functions.

<details><summary><b> Code </b></summary>

```c

// cpu.c

uint64_t cpu_load(CPU* cpu, uint64_t addr, uint64_t size) {
    return bus_load(&(cpu->bus), addr, size);
}

void cpu_store(CPU* cpu, uint64_t addr, uint64_t size, uint64_t value) {
    bus_store(&(cpu->bus), addr, size, value);
}

```

</details>

##### 4) **Instruction Decoding:**

Each RISC-V instruction is 32 bits wide and consists of different fields, including:

- **opcode**: Operation code (bits [6:0]).
- **rd**: Destination register (bits [11:7]).
- **rs1/rs2**: Source registers (bits [19:15] and [24:20]).
- **funct3/funct7**: Distinguish specific operations (e.g., addition vs. subtraction).
- **imm/shamt**: Immediate values and shift amounts for certain instructions.

Instructions are categorized into types such as **R-Type**, **I-Type**, **S-Type**, **B-Type**, **U-Type**, and **J-Type**, each with unique decoding rules. Specific decoding functions extract these fields for each type.

<details><summary><b> Code </b></summary>

```c

uint64_t rd(uint32_t inst) {
    return (inst >> 7) & 0x1f;  // rd in bits 11:7
}

uint64_t rs1(uint32_t inst) {
    return (inst >> 15) & 0x1f; // rs1 in bits 19:15
}

uint64_t rs2(uint32_t inst) {
    return (inst >> 20) & 0x1f; // rs2 in bits 24:20
}

uint64_t imm_I(uint32_t inst) {
    return ((int64_t)(int32_t)(inst & 0xfff00000)) >> 20;  // Immediate for I-type instructions
}

// Additional functions for other types (S-type, B-type, U-type, J-type)

```

</details>

##### 5) **Instruction Execution (`cpu_execute`):**

The `cpu_execute` function simulates the CPU's Arithmetic Logic Unit (ALU) and instruction decoder. It parses the fetched instruction and performs the operation based on the instruction type. For example:

- **ADD Immediate (`exec_ADDI`)**: Adds an immediate value to a register, ensuring proper sign extension.
- **Shift Instructions (`exec_SLLI`)**: Performs bitwise shifts using the shift amount.

Other instructions like **LUI** (Load Upper Immediate) and **AUIPC** (Add Upper Immediate to PC) handle upper immediate values, modifying either registers or the PC.

<details><summary><b> Code </b></summary>

```c

int cpu_execute(CPU *cpu, uint32_t inst) {
    int opcode = inst & 0x7f;           // Extract opcode
    int funct3 = (inst >> 12) & 0x7;    // Extract funct3
    int funct7 = (inst >> 25) & 0x7f;   // Extract funct7

    cpu->regs[0] = 0;  // Ensure register x0 remains 0

    switch (opcode) {
        case I_TYPE:
            switch (funct3) {
                case ADDI:  exec_ADDI(cpu, inst); break;
                case SLLI:  exec_SLLI(cpu, inst); break;
                case SLTI:  exec_SLTI(cpu, inst); break;
                case SLTIU: exec_SLTIU(cpu, inst); break;
                case XORI:  exec_XORI(cpu, inst); break;
                case SRI:
                    if (funct7 == SRLI) exec_SRLI(cpu, inst);
                    else if (funct7 == SRAI) exec_SRAI(cpu, inst);
                    break;
                case ORI:   exec_ORI(cpu, inst); break;
                case ANDI:  exec_ANDI(cpu, inst); break;
                default:    break;
            }
            break;
        default:
            fprintf(stderr,
                    "ERROR: Unsupported opcode:0x%x, funct3:0x%x, funct7:0x%x\n",
                    opcode, funct3, funct7);
            return 0;
    }
}

void exec_ADDI(CPU* cpu, uint32_t inst) {
    uint64_t imm = imm_I(inst);  // Extract immediate value
    cpu->regs[rd(inst)] = cpu->regs[rs1(inst)] + (int64_t)imm;  // Perform addition
    printf("Executed ADDI\n");
}

void exec_SLTI(CPU* cpu, uint32_t inst) {
    uint64_t imm = imm_I(inst);  // Extract immediate value
    cpu->regs[rd(inst)] = (cpu->regs[rs1(inst)] < (int64_t)imm) ? 1 : 0;  // Compare and set
    printf("Executed SLTI\n");
}

```

</details>

#### **Complete Simulator Loop**

The final step integrates all components into a functioning RISC-V simulator.

##### 1) **Loading the Binary File:**

The simulator reads the binary file containing the machine code and loads it into DRAM.

- A helper function `read_file()` handles file reading and transfers its contents to DRAM.
- The file’s content is stored starting at `DRAM_BASE`.

<details><summary><b> Code </b></summary>

```c

void read_file(CPU* cpu, char *filename) {
    FILE *file;
    uint8_t *buffer;
    unsigned long fileLen;

    // Open the file
    file = fopen(filename, "rb");
    if (!file) {
        fprintf(stderr, "Unable to open file %s", filename);
    }

    // Get the file size
    fseek(file, 0, SEEK_END);
    fileLen = ftell(file);
    fseek(file, 0, SEEK_SET);

    // Allocate memory for the file content
    buffer = (uint8_t *)malloc(fileLen + 1);
    if (!buffer) {
        fprintf(stderr, "Memory error!");
        fclose(file);
    }

    // Read file content into the buffer
    fread(buffer, fileLen, 1, file);
    fclose(file);

    // Print file content in hexadecimal format
    for (int i = 0; i < fileLen; i += 2) {
        if (i % 16 == 0) printf("\n%.8x: ", i);
        printf("%02x%02x ", *(buffer + i), *(buffer + i + 1));
    }
    printf("\n");

    // Copy binary file content to DRAM memory
    memcpy(cpu->bus.dram.mem, buffer, fileLen * sizeof(uint8_t));
    free(buffer);
}

```

</details>

##### 2) **Simulator Main Loop:**

The simulation runs in a loop, implementing a three-stage pipeline:

1. **Instruction Fetch:** The `cpu_fetch()` function reads the current instruction.
2. **Instruction Decode:** The instruction fields are extracted for further processing.
3. **Instruction Execute:** The `cpu_execute()` function performs the specified operation, updates the registers, and increments the `PC`.

The loop continues until either:

- The `PC` reaches 0, indicating the end of execution.
- An error occurs during instruction execution.

<details><summary><b> Code </b></summary>

```c

// Initialize CPU, registers, and program counter
struct CPU cpu;
cpu_init(&cpu);

// Read the input binary file
read_file(&cpu, argv[1]);

// Main CPU loop
while (1) {
    // Fetch the instruction
    uint32_t inst = cpu_fetch(&cpu);

    // Increment the program counter
    cpu.pc += 4;

    // Execute the instruction
    if (!cpu_execute(&cpu, inst))
        break;

    // Print register state for debugging
    dump_registers(&cpu);

    // Exit loop if the program counter is 0
    if(cpu.pc == 0)
        break;
}

return 0;

```

</details>

---

### Conclusion

In zkVM design, choosing an ISA like RISC-V or MIPS ensures compatibility and optimizes performance, memory, and finite field operations for zero-knowledge proofs. SP1, a RISC-V-compatible VM, excels with efficient proof generation, recursive proofs, and an accessible toolchain, making it ideal for blockchain and cryptographic use. The Rust-based RISC-V simulator offers better memory safety and performance, while the C version provides a simpler, low-level approach for efficient emulation. Together, these components create a robust environment for zkVM applications.

---

[memory-management](https://github.com/ETAAcademy/ETAAcademy-ZK-Meme/tree/main/Appendix/memory-management)

[rrs-main](https://github.com/ETAAcademy/ETAAcademy-ZK-Meme/tree/main/Appendix/rrs-main)

[riscv_emulator-main](https://github.com/ETAAcademy/ETAAcademy-ZK-Meme/tree/main/Appendix/riscv_emulator-main)

<div  align="center"> 
<img src="images/52_isa_sp1_riscv.gif" width="50%" />
</div>
