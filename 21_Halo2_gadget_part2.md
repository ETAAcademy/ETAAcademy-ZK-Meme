# ETAAcademy-ZKMeme: 21. Halo2 gadget

<table>
  <tr>
    <th>title</th>
    <th>tags</th>
  </tr>
  <tr>
    <td>21. Halo2 gadget</td>
    <td>
      <table>
        <tr>
          <th>zk-meme</th>
          <th>basic</th>
          <th>quick_read</th>
          <td>Halo2 gadget</td>
        </tr>
      </table>
    </td>
  </tr>
</table>

[Github](https://github.com/ETAAcademy)｜[Twitter](https://twitter.com/ETAAcademy)｜[ETA-ZK-Meme](https://github.com/ETAAcademy/ETAAcademy-ZK-Meme)

Authors: [Evta](https://twitter.com/pwhattie), looking forward to your joining

Constraint instance check for equality between assigned cells and specified values in a circuit. It covers accessing cells and columns within the circuit, considerations for enabling equality on columns, and integrating the constraint instance into the synthetic side of the design when assigning values to private inputs. Specifically, implementing an assigned cell to verify equality is crucial for ensuring accurate computations. Accessing cells within the circuit layout involves understanding absolute row numbers, which play a key role in applying constraints effectively. It's essential to manage circuit complexity by selectively enabling equality in layout constraints, balancing performance considerations. Enabling equality on all columns in circuit design entails trade-offs. While it simplifies certain operations, it can lead to increased complexity and potential performance issues, especially with permutations generating numerous columns. To optimize performance, it's advisable to disable equality on columns not involved in computations, reducing unnecessary constraints. Integrating a Fibonacci chip into the circuit requires meticulous planning. Considering previous values and private inputs ensures accurate computation. By carefully managing constraints and performance considerations, efficient circuit design can be achieved.

```rust
    pub fn expose_public(
        &self,
        mut layouter: impl Layouter<F>,
        cell: AssignedCell<F, F>,
        row: usize,
    ) -> Result<(), Error> {
        layouter.constrain_instance(cell.cell(), self.config.instance, row)
    }

```

The emergence of the "region.assign_advice_from_instance" API heralds a significant breakthrough in circuit design, offering developers a streamlined approach to setting up copy constraints. By facilitating the seamless transfer of data from absolute rows in the instance column to designated regions within the circuit, this API simplifies the setup process and enhances efficiency. Its versatility makes it adaptable to projects of all scales, while its intuitive design and automated functionality save developers valuable time and resources. As circuit design continues to evolve, it stands as a powerful tool, poised to meet the demands of increasingly complex projects and drive innovation in the field.

```rust
   pub fn assign(
        &self,
        mut layouter: impl Layouter<F>,
        nrows: usize,
    ) -> Result<AssignedCell<F, F>, Error> {
        layouter.assign_region(
            || "entire fibonacci table",
            |mut region| {
                self.config.selector.enable(&mut region, 0)?;
                self.config.selector.enable(&mut region, 1)?;

                let mut a_cell = region.assign_advice_from_instance(
                    || "1",
                    self.config.instance,
                    0,
                    self.config.advice,
                    0,
                )?;
                let mut b_cell = region.assign_advice_from_instance(
                    || "1",
                    self.config.instance,
                    1,
                    self.config.advice,
                    1,
                )?;

                for row in 2..nrows {
                    if row < nrows - 2 {
                        self.config.selector.enable(&mut region, row)?;
                    }

                    let c_cell = region.assign_advice(
                        || "advice",
                        self.config.advice,
                        row,
                        || a_cell.value().copied() + b_cell.value(),
                    )?;

                    a_cell = b_cell;
                    b_cell = c_cell;
                }

                Ok(b_cell)
            },
        )
    }
```

The zero gadget, designed to verify if a value is zero, incorporates various configurations and expressions to handle different scenarios efficiently. By utilizing the "q_enable" function and evaluating expressions such as "value_invert" and "one_minus_value_times_value_invert," developers can accurately determine the validity of input values. Through meticulous constraint application, the zero gadget ensures correct outputs while guarding against erroneous inputs. The conversation delves into the intricacies of setting up copy constraints and optimizing circuit performance, highlighting the API's significance in simplifying circuit design processes and enhancing computational efficiency.

```rust
    pub fn configure(
        meta: &mut ConstraintSystem<F>,
        q_enable: impl FnOnce(&mut VirtualCells<'_, F>) -> Expression<F>,
        value: impl FnOnce(&mut VirtualCells<'_, F>) -> Expression<F>,
        value_inv: Column<Advice>,
    ) -> IsZeroConfig<F> {
        let mut is_zero_expr = Expression::Constant(F::zero());

        meta.create_gate("is_zero", |meta| {
            //
            // valid | value |  value_inv |  1 - value * value_inv | value * (1 - value* value_inv)
            // ------+-------+------------+------------------------+-------------------------------
            //  yes  |   x   |    1/x     |         0              |  0
            //  no   |   x   |    0       |         1              |  x
            //  yes  |   0   |    0       |         1              |  0
            //  yes  |   0   |    y       |         1              |  0
            //
            let value = value(meta);
            let q_enable = q_enable(meta);
            let value_inv = meta.query_advice(value_inv, Rotation::cur());

            is_zero_expr = Expression::Constant(F::one()) - value.clone() * value_inv;
            vec![q_enable * value * is_zero_expr.clone()]
        });

        IsZeroConfig {
            value_inv,
            is_zero_expr,
        }
    }
```
