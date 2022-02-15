function tout=CompSum(x,t)
%Sum up the compatibility prediction of CompDecide
%x:tout of CompDecide;t:Actual compatibility mark;
[sizex,~]=size(x);
sum=0;tpos=0;fpos=0;tneg=0;fneg=0; %sum for total prediction number;t,f for true,false;pos,neg for positive(Compatible),negative(Incompatible)
for i=1:sizex
    if x(i)~=5
        sum=sum+1;
        if x(i)==0 && t(i)==0
            tpos=tpos+1;
        end
        if x(i)==0 && t(i)==10
            fpos=fpos+1;
        end
        if x(i)==10 && t(i)==10
            tneg=tneg+1;
        end
        if x(i)==10 && t(i)==0
            fneg=fneg+1;
        end
    end
end
tout=[sum;tpos;fpos;tneg;fneg];
end