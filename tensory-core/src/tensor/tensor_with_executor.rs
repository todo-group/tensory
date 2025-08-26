pub struct TensorWithExecutor<Id, T: TensorRepr, C> {
    tensor: Tensor<Id, T>,
    executor: C,
}

pub enum ContractExecutorError<Err> {
    MismatchedExecutors,
    Internal(Err),
}

trait Executor {
    fn exec_eq(&self, other: &Self) -> bool;
}

impl<Id, L: TensorRepr, Ex: Executor> Tensor<Id, L, Ex> {
    pub fn contract<R: TensorRepr>(self,rhs: Tensor<Id, R, Ex>) -> Result<Tensor<Id, Ex::Res, Ex>, ContractExecutorError<Ex::Err>> where Ex: ContractionContext<L, R> {
        self.executor.exec_eq(&rhs.executor).then(()).ok_or(ContractExecutorError::MismatchedExecutors).and()
    
        
        //map(|_|TensorMul::new(self, rhs).by(context).map_err(ContractExecutorError::Internal))
        todo!()
    }
}

impl<Id, L: TensorRepr, R: TensorRepr, Ex: ContractionContext<L, R>> Mul<Tensor<Id, R, Ex>>
    for Tensor<Id, L, Ex>
{
    type Output = TensorMul<Id, <Ex as ContractionContext<L, R>>::Res, Ex>;

    fn mul(self, rhs: Tensor<Id, R, Ex>) -> Self::Output {
        if self.executor != rhs.executor {
            panic!("Cannot multiply tensors with different executors");
        }

        TensorMul::new(self, rhs)
    }
}

impl<Id, A: TensorRepr, Ex: SvdContext<A>> Tensor<Id, A, Ex> {
    pub fn svd<'a>(self, u_legs: LegRefSet<'a, Id>) ->  {
        TensorSvd::new(self, u_legs)
    }
}
