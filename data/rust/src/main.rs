
/*
  Counts the number of digit characters, lowercase letters, and uppercase letters in a given string of length 8.
  
  Arguments:
  - s: A string of exactly 8 characters that may include digits, lowercase letters, and uppercase letters.

  Returns:
  - A tuple containing three usize values: the count of digit characters, the count of lowercase letters, and the count of uppercase letters in the string.

  Examples:
  - count_chars_types("yLOI2022") returns (4, 1, 3)
  - count_chars_types("IAKIOIOI") returns (0, 0, 8)
  - count_chars_types("1n2s0e1s") returns (4, 4, 0)
*/
fn count_chars_types(s: &str) -> (usize, usize, usize) {
    let mut digit_count = 0;
    let mut lower_count = 0;
    let mut upper_count = 0;

    for c in s.chars() {
        if c.is_digit(10) {
            digit_count += 1;
        } else if c.is_lowercase() {
            lower_count += 1;
        } else if c.is_uppercase() {
            upper_count += 1;
        }
    }

    (digit_count, lower_count, upper_count)
}

// Check function to verify the correctness of the generated function.




#[cfg(test)]
mod tests {
    use super::*;
 
    #[test]
    fn main() {
        assert_eq!(count_chars_types("yLOI2022"), (4, 1, 3));
        assert_eq!(count_chars_types("IAKIOIOI"), (0, 0, 8));
        assert_eq!(count_chars_types("1n2s0e1s"), (4, 4, 0));
    }
    

}