// Same counter with an injected bug: it wraps at 10 instead of 9, so the property fails.
module counter (
  input  logic       clk,
  input  logic       rst,
  output logic [3:0] cnt
);
  always_ff @(posedge clk) begin
    if (rst)               cnt <= 4'd0;
    else if (cnt == 4'd10) cnt <= 4'd0;
    else                   cnt <= cnt + 4'd1;
  end

`ifdef FORMAL
  initial assume (rst);
  always @(posedge clk) if (!rst) assert (cnt <= 4'd9);
`endif
endmodule
