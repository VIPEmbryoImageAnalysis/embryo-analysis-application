function out = linearRegression(y, z)
    % Calculate linear regression of the pixels over time
    out = polyfit(z, y, 1);
end