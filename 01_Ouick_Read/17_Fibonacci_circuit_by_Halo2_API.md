# ETAAcademy-ZKMeme: 17. Fibonacci circuit by Halo2 API

<table>
  <tr>
    <th>title</th>
    <th>tags</th>
  </tr>
  <tr>
    <td>17. Fibonacci circuit by Halo2 API.md</td>
    <td>
      <table>
        <tr>
          <th>zk-meme</th>
          <th>basic</th>
          <th>quick_read</th>
          <td>Fibonacci_circuit_by_Halo2_API.md</td>
        </tr>
      </table>
    </td>
  </tr>
</table>

[Github](https://github.com/ETAAcademy)｜[Twitter](https://twitter.com/ETAAcademy)｜[ETA-ZK-Meme](https://github.com/ETAAcademy/ETAAcademy-ZK-Meme)

Authors: [Eta](https://twitter.com/pwhattie), looking forward to your joining

There are three steps to implement a circuit: defining a config struct with column specifications, configuring a chip struct with gate definitions and assignment functions, and implementing a circuit trait to instantiate the circuit.

Halo 2 has different column types, including advice columns for private inputs, instance columns for public inputs, fixed columns for constants, and selector columns for controlling gates. Fixed columns and selectors are fixed across different inputs, while advice and instance columns vary. ConstraintSystem class is used to create columns, define custom gates, and enable permutation checks. Simple selectors are optimized for combining, while complex selectors are used in lookup arguments and are excluded from combining optimizations.

The `Config` type in a custom circuit acts as a flexible container for storing essential type information regarding the circuit's constraint system, such as column numbers, types, indices, and flags. While its structure is not enforced, it must implement `Clone`, and it's the Circuit implementer's job to translate its contents into the necessary format for the Layouter, typically involving arranging elements in a specific layout configuration.

```rust
#[derive(Debug, Clone)]
struct FiboChip<F: FieldExt> {
    config: FiboConfig,
    _marker: PhantomData<F>,
}
```

`meta.query_advice` or`meta.query_selector` functions generate an expression, which is essentially a reference to a cell in the matrix. For example, a query for advice cells, contains information like the query index, column index, and rotation. These expressions act as symbolic variables that can be used in polynomial constraints. The `meta.create_gate` function takes a function from virtual cells to constraints, adding these constraints to the accumulator of `meta`.

```rust
impl<F: FieldExt> FiboChip<F> {
    pub fn construct(config: FiboConfig) -> Self
        Self {
            config,
            _marker: PhantomData,
        }
    }

    pub fn configure(
        meta: &mut ConstraintSystem<F>,
        advice: [Column<Advice>; 3],
        instance: Column<Instance>,
    ) -> FiboConfig {
        let col_a = advice[0];
        let col_b = advice[1];
        let col_c = advice[2];
        let selector = meta.selector();

        meta.enable_equality(col_a);
        meta.enable_equality(col_b);
        meta.enable_equality(col_c);
        meta.enable_equality(instance);

        meta.create_gate("add", |meta| {
            //
            // col_a | col_b | col_c | selector
            //   a      b        c       s
            //
            let s = meta.query_selector(selector);
            let a = meta.query_advice(col_a, Rotation::cur());
            let b = meta.query_advice(col_b, Rotation::cur());
            let c = meta.query_advice(col_c, Rotation::cur());
            vec![s * (a + b - c)]
        })
```

`Circuit` trait, which is utilized in various stages such as prover and verifier key generation, as well as actual proving and verifying processes. This trait includes functions for initializing without witnesses, configuring gates, and synthesizing the circuit layout. Implementing this trait involves three main functions: first, setting up the circuit without witness values; second, configuring it, which essentially defines the constraint system using a mutable ConstraintSystem and returns a custom associated type `Config`; and third, synthesizing the circuit by writing to a provided Layouter based on the configuration provided. The `configure` function fills the constraint system with the required constraints, while the `synthesize` method combines these constraints and assigns an absolute layout. Overall, `configure` establishes a concrete but relative layout, while `synthesize` finalizes it by assigning absolute positions.

```rust
#[derive(Default)]
struct MyCircuit<F> {
    pub a: Value<F>,
    pub b: Value<F>,
}

impl<F: FieldExt> Circuit<F> for MyCircuit<F> {
    type Config = FiboConfig;
    type FloorPlanner = SimpleFloorPlanner;

    fn without_witnesses(&self) -> Self {
        Self::default()
    }

    fn configure(meta: &mut ConstraintSystem<F>) -> Self::Config {
        let col_a = meta.advice_column();
        let col_b = meta.advice_column();
        let col_c = meta.advice_column();
        let instance = meta.instance_column();
        FiboChip::configure(meta, [col_a, col_b, col_c], instance)
    }

    fn synthesize(
        &self,
        config: Self::Config,
        mut layouter: impl Layouter<F>,
    ) -> Result<(), Error> {
        let chip = FiboChip::construct(config);

        let (prev_a, mut prev_b, mut prev_c) =
            chip.assign_first_row(layouter.namespace(|| "first row"), self.a, self.b)?;

        chip.expose_public(layouter.namespace(|| "private a"), &prev_a, 0)?;
        chip.expose_public(layouter.namespace(|| "private b"), &prev_b, 1)?;

        for _i in 3..10 {
            let c_cell = chip.assign_row(layouter.namespace(|| "next row"), &prev_b, &prev_c)?;
            prev_b = prev_c;
            prev_c = c_cell;
        }

        chip.expose_public(layouter.namespace(|| "out"), &prev_c, 2)?;

        Ok(())
    }
}
```

```

```
