
let absoluteValue (n: int): int =
    if n < 0 then -n
    else n


let check () =
    if absoluteValue -10 <> 10 then
        failwith "Test Case 1 failed"
    if absoluteValue 5 <> 5 then
        failwith "Test Case 2 failed"
    if absoluteValue 0 <> 0 then
        failwith "Test Case 3 failed"
    if absoluteValue -10000 <> 10000 then
        failwith "Test Case 4 failed"
    if absoluteValue 9999 <> 9999 then
        failwith "Test Case 5 failed"
    if absoluteValue -1 <> 1 then
        failwith "Test Case 6 failed"

check ()