use ndarray::{Array1, ArrayView1};

/// Greedy split offsets for groups summing to no more than `max_value`.
/// Byte-identical to the numba `splits_sum_le_value` in `_utils.py`.
pub fn splits_sum_le_value(arr: ArrayView1<i64>, max_value: f64) -> Array1<i64> {
    let mut indices: Vec<i64> = vec![0];
    let mut current_sum: i64 = 0;
    for (idx, &value) in arr.iter().enumerate() {
        current_sum += value;
        if current_sum as f64 > max_value {
            indices.push(idx as i64);
            current_sum = value;
        }
    }
    indices.push(arr.len() as i64);
    Array1::from(indices)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn docstring_example() {
        // splits_sum_le_value([5,5,11,9,2,7], 10) -> [0,2,3,4,6]
        let a = array![5_i64, 5, 11, 9, 2, 7];
        assert_eq!(splits_sum_le_value(a.view(), 10.0), array![0_i64, 2, 3, 4, 6]);
    }

    #[test]
    fn empty_array() {
        let a: Array1<i64> = Array1::from(vec![]);
        assert_eq!(splits_sum_le_value(a.view(), 10.0), array![0_i64, 0]);
    }

    #[test]
    fn single_over_max_kept_in_own_group() {
        let a = array![3_i64, 100, 3];
        // 3<=10; +100 ->103>10 push 1 reset 100; +3=103>10 push 2 reset 3; end push 3
        assert_eq!(splits_sum_le_value(a.view(), 10.0), array![0_i64, 1, 2, 3]);
    }
}
