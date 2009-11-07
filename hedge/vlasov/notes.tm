<TeXmacs|1.0.7.2>

<style|<tuple|generic|maxima>>

<\body>
  <\equation*>
    <big|int><rsub|-c<rsub|0>><rsup|c<rsub|0>>f(v)\<mathd\>v=<big|int><rsub|-\<infty\>><rsup|\<infty\>>f(v(p))v<rprime|'>(p)\<mathd\>p
  </equation*>

  <\session|maxima|default>
    <\output>
      \;

      Maxima 5.17.1 http://maxima.sourceforge.net

      Using Lisp GNU Common Lisp (GCL) GCL 2.6.7 (aka GCL)

      Distributed under the GNU Public License. See the file COPYING.

      Dedicated to the memory of William Schelter.

      The function bug_report() provides bug reporting information.
    </output>

    <\unfolded-io>
      <with|color|red|(<with|math-font-family|rm|%i>3) <with|color|black|>>
    <|unfolded-io>
      diff(c*p/sqrt(p^2+c^2*m^2),p)
    <|unfolded-io>
      <with|mode|math|math-display|true|<with|mode|text|font-family|tt|color|red|(<with|math-font-family|rm|%o3>)
      <with|color|black|>><frac|c|<sqrt|p<rsup|2>+c<rsup|2>*m<rsup|2>>>-<frac|c*p<rsup|2>|<left|(>p<rsup|2>+c<rsup|2>*m<rsup|2><right|)><rsup|<frac|3|2>>>>
    </unfolded-io>

    <\input>
      <with|color|red|(<with|math-font-family|rm|%i>4) <with|color|black|>>
    <|input>
      \;
    </input>
  </session>
</body>

<\initial>
  <\collection>
    <associate|language|german>
    <associate|page-type|letter>
  </collection>
</initial>