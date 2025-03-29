pub mod matmult;

mod tests {
    use csv::Writer;
    use util::algebra::field::mersenne61_ext::Mersenne61Ext;

    use crate::matmult::{mat_mult, naive_mat_mult, naive_opening, Matrix};

    #[test]
    fn test_mat_mult() {
        // let row_size = vec![150, 300, 600, 900, 1200];
        let row_size = vec![150, 300, 600, 900, 1200];
        let mut wtr = Writer::from_path("mat_mult.csv").unwrap();
        for r in row_size {
            let mat_a = Matrix::<Mersenne61Ext>::sample(r, 768);
            let mat_b = Matrix::<Mersenne61Ext>::sample(768, 2304);
            let mat_c = mat_a.clone() * mat_b.clone();

            println!("size {} start", r);
            let size = mat_mult(&mat_a, &mat_b, &mat_c);
            wtr.write_record(&[r.to_string(), size.to_string()])
                .unwrap();
        }
    }

    #[test]
    fn test_naive_mat_mult() {
        let row_size = vec![150, 300, 600, 900, 1200];
        // let row_size = vec![900];
        let mut wtr = Writer::from_path("naive_mat_mult.csv").unwrap();
        for r in row_size {
            let mat_a = Matrix::<Mersenne61Ext>::sample(r, 768);
            let mat_b = Matrix::<Mersenne61Ext>::sample(768, 2304);
            let mat_c = mat_a.clone() * mat_b.clone();

            let size = naive_mat_mult(&mat_a, &mat_b, &mat_c);
            wtr.write_record(&[r.to_string(), size.to_string()])
                .unwrap();
        }
    }

    #[test]
    fn test_naive_opening() {
        let row_size = vec![150, 300, 600, 900, 1200];
        // let row_size = vec![900];
        let mut wtr = Writer::from_path("naive_opening.csv").unwrap();
        for r in row_size {
            let mat_a = Matrix::<Mersenne61Ext>::sample(r, 768);
            let mat_b = Matrix::<Mersenne61Ext>::sample(768, 2304);
            let mat_c = mat_a.clone() * mat_b.clone();

            let size = naive_opening(&mat_a, &mat_b, &mat_c);
            wtr.write_record(&[r.to_string(), size.to_string()])
                .unwrap();
        }
    }
}
