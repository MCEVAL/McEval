Module Module1
Function CalculateExpression(ByVal a As Integer, ByVal b As Integer, ByVal c As Integer) As Integer
    ' Calculates the value of the expression (a+b) * c
    Return (a + b) * c
End Function

    Sub Main()
        ' Test the CalculateExpression function
        Debug.Assert(CalculateExpression(2, 3, 5) = 25)
        Debug.Assert(CalculateExpression(-1, 2, 3) = 3)
        Debug.Assert(CalculateExpression(0, 0, 1) = 0)
        Debug.Assert(CalculateExpression(10, -5, 2) = 10)
        Debug.Assert(CalculateExpression(-2, -3, -4) = 20)
        Debug.Assert(CalculateExpression(1000, 2000, 3) = 9000)
        Debug.Assert(CalculateExpression(-100, 50, 10) = -500)
    End Sub
End Module