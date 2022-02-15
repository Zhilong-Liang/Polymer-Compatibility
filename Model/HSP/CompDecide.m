function tout=CompDecide(n1,t1,n0,t0)
%Decide whether blends in [n1 t1] is compatible, using HSP values in [n0 t0]
%n1:[Polymer1Name Polymer2Name];t1:[Polymer1wt% Polymer2wt%]
%n0:[PolymerName];t0:[HSP(cal/cc) mw(g/mol) AccurateDensity(g/cc)]
%tout:[CompatibilityMark]

[size1,~]=size(n1);[size0,~]=size(n0);tout=zeros(size1,2); %Initialize, marked as 0 for COMPATIBLE
for t=1:size1
  j1=0;j2=0;
  name1=n1(t,1);name2=n1(t,2);
  for j=1:size0
    name0=n0(j);
    if strcmp(name1,name0)
      j1=j;
    end
    if strcmp(name2,name0)
      j2=j;
    end
  end
  if j1==0 || j2==0 || isnan(t0(j1,1)) || isnan(t0(j2,1)) || t0(j1,1)==0 || t0(j2,1)==0    %Find if HSP value is NOT present in n0 and t0
    tout(t)=5;     %Marked as 5 for UNCERTAIN
  else
    w1=t1(t,1)/100;w2=1-w1;m1=t0(j1,2);rou1=t0(j1,3);m2=t0(j2,2);rou2=t0(j2,3);
    hm=sqrt(w1*m1*rou1*(t0(j1,1)-t0(j2,1))^2*(w2/(w1*m2*rou2+w2*m1*rou1))^2);
    tout(t,2)=hm;  %Save 
    if hm>=0.010
      tout(t)=10;  %Marked as 10 for INCOMPATIBLE
    end
  end
end

end
  
