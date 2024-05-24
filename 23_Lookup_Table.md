# ETAAcademy-ZKMeme: 23. Lookup Table

<table>
  <tr>
    <th>title</th>
    <th>tags</th>
  </tr>
  <tr>
    <td>23. Lookup Table</td>
    <td>
      <table>
        <tr>
          <th>zk-meme</th>
          <th>basic</th>
          <th>quick_read</th>
          <td>Lookup Table</td>
        </tr>
      </table>
    </td>
  </tr>
</table>

[Github](https://github.com/ETAAcademy)｜[Twitter](https://twitter.com/ETAAcademy)｜[ETA-ZK-Meme](https://github.com/ETAAcademy/ETAAcademy-ZK-Meme)

Authors: [Eta](https://twitter.com/pwhattie), looking forward to your joining

Basic range check ensure that a given value stays within a specified range within the circuit. Additionally, expanding this range check configuration to include a lookup argument, beneficial for efficiently verifying larger ranges while maintaining low constraint degrees.

How to configure a range check mechanism within a constraint system? The process involves passing advice columns as parameters, which are often shared across different configurations, creating a selector for toggling the range check constraint and then defining the configuration for the range check gate. This gate ensures that a value
v stays within a specified range R within the circuit, e.g. $v * (1 - v) * (2 - v) * ... * (R - 1 - v) = 0$. The process includes querying the selector and advice columns, then using a closure to generate the expression for the range check. `Constraints::with_selector` API which simplifies the process of returning expressions by abstracting the selector implementation. By specifying one selector, the API automatically multiplies each expression by that selector behind the scenes. Finally, it involves iterating through allowed values in the range and adding factors to the expression accordingly.

```rust
    pub fn configure(meta: &mut ConstraintSystem<F>, value: Column<Advice>) -> Self {
        let q_range_check = meta.selector();

        meta.create_gate("range check", |meta| {
            //        value     |    q_range_check
            //       ------------------------------
            //          v       |         1

            let q = meta.query_selector(q_range_check);
            let value = meta.query_advice(value, Rotation::cur());

            // Given a range R and a value v, returns the expression
            // (v) * (1 - v) * (2 - v) * ... * (R - 1 - v)
            let range_check = |range: usize, value: Expression<F>| {
                assert!(range > 0);
                (1..range).fold(value.clone(), |expr, i| {
                    expr * (Expression::Constant(F::from(i as u64)) - value.clone())
                })
            };

            Constraints::with_selector(q, [("range check", range_check(RANGE, value))])
        });

        Self {
            q_range_check,
            value,
            _marker: PhantomData,
        }
    }
```

In the creating expressions within a circuit, selectors are necessary for toggling range check constraints and configuring the range check gate. The ultimate goal is to establish a seamless process where configurations act as templates, reducing the developer's workload. Finally, the method for assigning values to the gate is elaborated upon, emphasizing the importance of enabling key range checks and specifying offsets for the assigned values.

```rust
    pub fn assign(
        &self,
        mut layouter: impl Layouter<F>,
        value: Value<Assigned<F>>,
    ) -> Result<RangeConstrained<F, RANGE>, Error> {
        layouter.assign_region(
            || "Assign value",
            |mut region| {
                let offset = 0;

                // Enable q_range_check
                self.q_range_check.enable(&mut region, offset)?;

                // Assign value
                region
                    .assign_advice(|| "value", self.value, offset, || value)
                    .map(RangeConstrained)
            },
        )
    }
```

Next, a lookup argument, in the layout to run parallel with the simple expression, is controlled by a different selector called the lookup selector. It utilizes a lookup table containing all the values within the desired range. When using key range check, `q_lookup` is not needed, unless both are to be enabled simultaneously. The process involves defining a sub-module table.rs to contain the lookup table, specifying the number of bits and the range of values.

The load function assigns all fixed values to the table during generation time. It takes a layouter as input and iterates through each value based on the specified number of bits, assigning each value to a cell in the table. This assignment occurs within a layouter's assign table method, which is a specialized API designed specifically for lookup tables. Once the load function is defined, table configuration is implemented to define a configure function to instantiate a table config. Finally, the completed load and configure functions can be utilized in the overall implementation.

```rust
#[derive(Debug, Clone)]
struct RangeCheckConfig<F: FieldExt, const RANGE: usize, const LOOKUP_RANGE: usize> {
    q_range_check: Selector,
    q_lookup: Selector,
    value: Column<Advice>,
    table: RangeTableConfig<F, LOOKUP_RANGE>,
}
```

```rust
#[derive(Debug, Clone)]
pub(super) struct RangeTableConfig<F: FieldExt, const RANGE: usize> {
    pub(super) value: TableColumn,
    _marker: PhantomData<F>,
}

impl<F: FieldExt, const RANGE: usize> RangeTableConfig<F, RANGE> {
    pub(super) fn configure(meta: &mut ConstraintSystem<F>) -> Self {
        let value = meta.lookup_table_column();

        Self {
            value,
            _marker: PhantomData,
        }
    }

    pub(super) fn load(&self, layouter: &mut impl Layouter<F>) -> Result<(), Error> {
        layouter.assign_table(
            || "load range-check table",
            |mut table| {
                let mut offset = 0;
                for value in 0..RANGE {
                    table.assign_cell(
                        || "num_bits",
                        self.value,
                        offset,
                        || Value::known(F::from(value as u64)),
                    )?;
                    offset += 1;
                }

                Ok(())
            },
        )
    }
}
```
