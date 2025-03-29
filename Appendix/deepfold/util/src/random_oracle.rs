use crate::algebra::field::MyField;
use rand::Rng;

#[derive(Debug, Clone)]
pub struct RandomOracle<T: MyField> {
    pub beta: T,
    pub rlc: T,
    pub folding_challenges: Vec<T>,
    pub deep: Vec<T>,
    pub alpha: Vec<T>,
    pub query_list: Vec<usize>,
}

impl<T: MyField> RandomOracle<T> {
    pub fn new(total_round: usize, query_num: usize) -> Self {
        RandomOracle {
            beta: T::random_element(),
            rlc: T::random_element(),
            folding_challenges: (0..total_round)
                .into_iter()
                .map(|_| T::random_element())
                .collect(),
            deep: (0..total_round)
                .into_iter()
                .map(|_| T::random_element())
                .collect(),
            alpha: (0..total_round)
                .into_iter()
                .map(|_| T::random_element())
                .collect(),
            query_list: (0..query_num)
                .into_iter()
                .map(|_| rand::thread_rng().gen())
                .collect(),
        }
    }
}
