using System;
using System;
using System.Collections.Generic;
using System.Diagnostics;

class Program
{


    /* Counts the number of digit, lowercase, and uppercase characters in a given string of length 8.
        >>> CountCharacters("1n2s0e1s")
        (4, 4, 0)
    */

    static (int, int, int) CountCharacters(string s)
{
        if (s.Length != 8) {
            throw new ArgumentException("String must be exactly 8 characters long.");
        }

        int digits = 0, lowercase = 0, uppercase = 0;
        foreach (char c in s) {
            if (char.IsDigit(c)) digits++;
            else if (char.IsLower(c)) lowercase++;
            else if (char.IsUpper(c)) uppercase++;
        }

        return (digits, lowercase, uppercase);
    }

    // Check function to verify the correctness of the CountCharacters function.
    public static void Check() {
        var result1 = CountCharacters("1n2s0e1s");
        Debug.Assert(result1 == (4, 4, 0), "Test case 1 failed");

        var result2 = CountCharacters("AbCdEfG");
        Debug.Assert(result2 == (0, 0, 8), "Test case 2 failed");

        var result3 = CountCharacters("12345678");
        Debug.Assert(result3 == (8, 0, 0), "Test case 3 failed");

        Console.WriteLine("All test cases passed!");
    }
    static void Main()
    {
        Debug.Assert(CountCharacters("yLAX2022") == (4, 1, 3));
        Debug.Assert(CountCharacters("MBKKOKOK") == (0, 0, 8));
        Debug.Assert(CountCharacters("1n2s0e1s") == (4, 4, 0));
        Debug.Assert(CountCharacters("1234ABCD") == (4, 0, 4));


    }
}